from __future__ import annotations

import errno
import fcntl
import functools
import io
import logging
import operator
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import anyio
from gi.repository import GLib  # type: ignore

from ._dbus_facade import DBusFacade, DBusUnavailable

# References:
# https://docs.kernel.org/admin-guide/abi-stable.html#


@dataclass(frozen=True)
class BacklightDevice:
    name: str | None
    serial: str | None
    path: Path
    i2c_num: int
    card_name: str | None
    connected: bool
    mul: float
    max_level: float | int

    def matches(self, other: BacklightDevice | str | int | None):
        ret = False
        if other is not None:
            if isinstance(other, BacklightDevice):
                ret = self == other
            elif isinstance(other, str):
                for s in (self.name, self.serial, self.card_name):
                    if s is not None and (ret := s.casefold() == other.casefold()):
                        break
            elif isinstance(other, int):
                ret = self.i2c_num == other

        return ret


class BacklightInterface:
    EDID_NAME_DESC = b"\0\0\0\xfc\0"
    EDID_SN_DESC = b"\0\0\0\xff\0"
    PATH_DRM = "/sys/class/drm"
    GLOB_I2C_DEV = "i2c-*"
    GLOB_DRM_CARD = "card*"
    PATH_DDC_DEV = "ddc/i2c-dev"
    PATH_DEV = "/dev"
    API_PAUSE = 0.06  # time to wait between packets, the spec says 30 ms
    I2C_ADDR_TX = 0x37  # packet transmission base address
    BUFFSZ_EDID = 128

    EDID_FORMAT: str = (
        ">"  # big-endian
        "18s"  # display descriptor block 1
        "18s"  # display descriptor block 2
        "18s"  # display descriptor block 3
        "18s"  # display descriptor block 4
    )

    def __init__(self, disp: str | int | None = None):
        self.disp = disp
        self.unpack = struct.Struct(self.EDID_FORMAT).unpack
        self.device: BacklightDevice | None = None
        self.fd = -1
        self.rd_handle = None
        self.wr_handle = None

    async def fd_write_async(self, data: bytes | bytearray):
        if self.wr_handle is None:
            wr_file_obj = io.FileIO(self.fd, "w", closefd=False)
            self.wr_handle = anyio.wrap_file(wr_file_obj)

        if (nwritten := await self.wr_handle.write(data)) < len(data):
            raise IOError("not all data is written")
        return nwritten

    async def fd_read_async(self, length: int) -> bytes:
        if self.rd_handle is None:
            rd_file_obj = io.FileIO(self.fd, "r", closefd=False)
            self.rd_handle = anyio.wrap_file(rd_file_obj)

        return await self.rd_handle.read(length)

    @classmethod
    def i2c_to_drm(cls, dev_num: int) -> str | None:
        ret = None

        if dev_num >= 0:
            i2c_name = "i2c-" + str(dev_num)

            if (Path(cls.PATH_DEV) / i2c_name).exists():
                for path_card in Path(cls.PATH_DRM).glob(cls.GLOB_DRM_CARD):
                    if (path_card / cls.PATH_DDC_DEV / i2c_name).is_dir():
                        ret = path_card.name
                        break
        return ret

    @classmethod
    def drm_to_i2c(cls, card_name: str) -> int:

        ret = -1

        if card_name is not None:
            i2c_dev_path = Path(cls.PATH_DRM) / card_name / cls.PATH_DDC_DEV

            for path_i2c in i2c_dev_path.glob(cls.GLOB_I2C_DEV):
                if (Path(cls.PATH_DEV) / path_i2c.name).exists():
                    try:
                        ret = int(path_i2c.name.split("-")[-1])
                    except ValueError:
                        pass
                    break
        return ret

    @classmethod
    async def detect_connected_drm(cls):

        ret = []

        for path_card in Path(cls.PATH_DRM).glob(cls.GLOB_DRM_CARD):
            connected = False
            path_status = path_card / "status"

            if path_status.is_file():
                try:
                    async with await anyio.open_file(path_status, "w") as fhandle:
                        await fhandle.write("detect")
                        await fhandle.flush()

                    await anyio.sleep(cls.API_PAUSE)
                except OSError as ex:
                    if ex.errno == errno.EACCES:
                        logging.debug(
                            "%s: cannot force status detection", path_card.name
                        )
                    else:
                        raise

                async with await anyio.open_file(path_status, "r") as fhandle:
                    if (await fhandle.readline()).strip() == "connected":
                        connected = True

            i2c_num = cls.drm_to_i2c(path_card.name)

            ret.append((path_card.name, i2c_num, connected))

        return ret

    def get_dev_id(self, edid: bytes | str) -> tuple[str | None, str | None]:

        name = None
        serial = None
        blocks = b""

        # https://en.wikipedia.org/wiki/Extended_Display_Identification_Data
        if isinstance(edid, str):
            edid = bytes.fromhex(edid)

        if len(edid) >= 128:
            try:
                blocks = self.unpack(edid[54:126])
            except struct.error as e:
                raise ValueError("cannot unpack EDID") from e

            for desc_blk in blocks:
                if desc_blk.startswith(self.EDID_NAME_DESC):
                    name_bytes = desc_blk[len(self.EDID_NAME_DESC) :]
                    name = name_bytes.decode("ascii", "replace").strip()
                elif desc_blk.startswith(self.EDID_SN_DESC):
                    sn_bytes = desc_blk[len(self.EDID_NAME_DESC) :]
                    serial = sn_bytes.decode("ascii", "replace").strip()

                if name is not None and serial is not None:
                    break

        return (name, serial)

    async def init(self):
        raise NotImplementedError()

    async def get_level(self) -> float:
        raise NotImplementedError()

    async def set_level(self, value: int | float):
        raise NotImplementedError()

    async def change_level(self, delta: int | float) -> float:
        curr_level = await self.get_level()
        await anyio.lowlevel.checkpoint()
        level = curr_level + delta
        await self.set_level(level)
        return level

    async def close(self):
        self.device = None

        # the handles do not own the file descriptor
        if self.rd_handle is not None:
            await self.rd_handle.aclose()
            self.rd_handle = None

        if self.wr_handle is not None:
            await self.wr_handle.aclose()
            self.wr_handle = None

        if self.fd >= 0:
            os.close(self.fd)
        self.fd = -1
        await anyio.lowlevel.checkpoint()

    async def __aenter__(self) -> BacklightInterface:
        await self.init()
        return self

    async def __aexit__(self, ex_type, ex_value, ex_traceback):
        return await self.close()


class LogindSession(DBusFacade):
    BASE_SVC = "org.freedesktop.login1"
    SESSION_OBJ = "/org/freedesktop/login1/session/auto"
    SESSION_IFACE = "org.freedesktop.login1.Session"

    def set_brightness(self, device: str, level: int):
        """Sets the brightness of a backlight device without write access to
        it. Blocks, call it in a worker thread."""
        self.call(
            self.SESSION_OBJ,
            self.SESSION_IFACE,
            "SetBrightness",
            GLib.Variant("(ssu)", ("backlight", device, level)),
        )


class BacklightACPI(BacklightInterface):
    PATH_BL = "/sys/class/backlight"
    PATH_I2C_BUS = "/sys/bus/i2c/devices/"
    GLOB_DRM_EDID = "card*/edid"
    GLOB_DRM_BR = "ddc/**/*backlight*/**/actual_brightness"

    def __init__(self, disp: str | int | None = None):
        super().__init__(disp)
        self.device = None
        self.fd = -1
        # -1 if brightness is set by logind
        self.wr_fd: int | None = None

    async def _get_scale_sync(self, dir_bl: Path | str) -> tuple[int, float]:
        mul = 0.0
        max_level = 100

        if isinstance(dir_bl, str):
            dir_bl = Path(dir_bl)

        try:
            async with await anyio.open_file(dir_bl / "max_brightness", "r") as fhandle:
                max_level = int((await fhandle.readline()).strip())
                mul = max_level / 100
        except (FileNotFoundError, TypeError, ValueError):
            pass

        return max_level, mul

    async def _get_display(self) -> BacklightDevice:

        ret = None

        card_glob = "card*"

        drm_cards = await self.detect_connected_drm()

        if self.disp is not None:
            if isinstance(self.disp, str):
                for card_name, i2c_num, connected in drm_cards:
                    if self.disp.strip().casefold() == card_name.strip().casefold():
                        card_glob = card_name
                        break
            elif isinstance(self.disp, int):
                for card_name, i2c_num, connected in drm_cards:
                    if self.disp == i2c_num:
                        card_glob = card_name
                        break

        disconnected = []

        # Try to get display information and backlight multiplier directly from
        # DRM sysfs.
        for f_edid in Path(self.PATH_DRM).glob(card_glob + "/edid"):
            name, serial = None, None

            async with await anyio.open_file(f_edid, "rb") as fhandle:
                edid_bytes = await fhandle.read(self.BUFFSZ_EDID)

                name, serial = self.get_dev_id(edid_bytes)

            if name is not None or serial is not None:
                card_dir = Path(f_edid).resolve().parent

                for f_bl in card_dir.glob(self.GLOB_DRM_BR):
                    dir_bl = f_bl.resolve().parent

                    max_level, mul = await self._get_scale_sync(dir_bl)

                    if mul > 0:
                        i2c_num = -1
                        is_connected = False

                        for card_name, _i2c_num, connected in drm_cards:
                            if card_name == card_dir.name:
                                i2c_num = _i2c_num
                                is_connected = connected
                                break

                        dev = BacklightDevice(
                            name,
                            serial,
                            dir_bl,
                            i2c_num,
                            card_dir.name,
                            is_connected,
                            mul,
                            max_level,
                        )

                        if self.disp is None or dev.matches(self.disp):
                            if not dev.connected:
                                disconnected.append(dev)
                            else:
                                ret = dev
                                break
            await anyio.lowlevel.checkpoint()

        if ret is None:
            for dentry in os.scandir(self.PATH_BL):
                if dentry.is_dir():
                    path_bl = Path(dentry.path).resolve()

                    if self.disp is None or self.disp == path_bl.name:
                        max_level, mul = await self._get_scale_sync(path_bl)

                        if mul > 0:
                            dev = BacklightDevice(
                                path_bl.name,
                                None,
                                path_bl,
                                -1,
                                None,
                                False,
                                mul,
                                max_level,
                            )
                            ret = dev
                            break
                await anyio.lowlevel.checkpoint()

        if ret is None and disconnected:
            ret = disconnected[0]

        return ret

    async def init(self):
        device = await self._get_display()
        await self.close()

        if device is None:
            raise OSError(
                errno.ENODEV, f"{self.disp}: no compatible backlight device found"
            )
        else:
            self.device = device

        if self.fd < 0:
            # writing usually needs a udev rule, logind is used otherwise
            self.fd = os.open(self.device.path / "brightness", os.O_RDONLY)

    def _get_raw(self) -> int:
        buff = os.pread(self.fd, 16, 0)
        return int(buff.decode("ascii").strip() or 0)

    async def _set_raw(self, raw: int):
        if self.device is None:
            return

        raw = max(0, min(raw, int(self.device.max_level)))

        if self.wr_fd is None:
            try:
                self.wr_fd = os.open(self.device.path / "brightness", os.O_WRONLY)
            except PermissionError:
                self.wr_fd = -1

        if self.wr_fd >= 0:
            os.pwrite(self.wr_fd, str(raw).encode(), 0)
            return

        try:
            await anyio.to_thread.run_sync(
                LogindSession().set_brightness, self.device.path.name, raw
            )
        except (GLib.Error, DBusUnavailable) as ex:
            # e.g. no logind, or an inactive session: the brightness stays
            logging.warning(
                "%s: cannot set the brightness, no write access and logind failed: %s",
                self.device.path.name,
                getattr(ex, "message", ex),
            )

    async def get_level(self) -> float:
        if self.device is None:
            return 0.0
        return self._get_raw() / self.device.mul

    async def set_level(self, value: int | float):
        if self.device is not None:
            await self._set_raw(round(value * self.device.mul))

    async def change_level(self, delta: int | float) -> float:
        """Steps by at least one hardware level, panels may have as few as
        ten. Returns the new level."""
        if self.device is None:
            return 0.0

        raw = self._get_raw()
        target = round(raw + delta * self.device.mul)

        if target == raw and delta != 0:
            target = raw + (1 if delta > 0 else -1)

        target = max(0, min(target, int(self.device.max_level)))
        await self._set_raw(target)

        # what the device shows, the change may have failed
        return self._get_raw() / self.device.mul

    async def close(self):
        if self.wr_fd is not None and self.wr_fd >= 0:
            os.close(self.wr_fd)
        self.wr_fd = None
        await super().close()


class BacklightDDCCI(BacklightInterface):
    # see https://boichat.ch/nicolas/ddcci/specs.html
    API_PAUSE_MIN = 0.03
    BUFFSZ_READ = 8
    I2C_ADDR_EDID = 0x50  # this is an address not an index, use ioctl on it
    I2C_CHG = 0x03  # change a value command
    I2C_CKSUM_XOR = 0x50
    I2C_IDX_BL = 0x10  # brightness block index
    I2C_IDX_EDID = 0x00
    I2C_PRE_VALUE = 0x02  # following value marker
    I2C_READ = 0x01  # read a value command
    I2C_SLAVE = 0x0703  # use slave address
    I2C_SRC_ADDR = 0x6E  # ACK of the packet previously written
    I2C_WR_LEN = 0x80  # length mask
    I2C_WR_SUB = 0x51  # sub-address

    PATH_I2C_ADAPTERS = "/sys/bus/i2c/devices"
    # adapters of memory modules, sensors, touchpads, never of displays
    NON_DISPLAY_ADAPTERS = ("smbus", "designware", "smu", "soc:i2cdsi")

    def __init__(self, disp: int | str | None = None):
        super().__init__(disp)

        self.device = None
        self.fd = -1

    @classmethod
    def _candidate_buses(cls, drm_cards: list[tuple[str, int, bool]]) -> list[Path]:
        """I2C buses that may connect a display: DDC buses of DRM connectors
        first, then other adapters, except known non-display ones. Probing
        writes to the bus, which may confuse other devices on it, e.g. memory
        module EEPROMs share the EDID address."""
        buses = [
            Path(cls.PATH_DEV) / f"i2c-{i2c_num}"
            for i2c_num in sorted(
                {i2c_num for _, i2c_num, _ in drm_cards if i2c_num >= 0}
            )
        ]

        # e.g. the NVIDIA driver does not link DDC buses to DRM connectors
        for i2c_dev in sorted(Path(cls.PATH_DEV).glob(cls.GLOB_I2C_DEV)):
            if i2c_dev in buses:
                continue

            try:
                adapter = (
                    Path(cls.PATH_I2C_ADAPTERS) / i2c_dev.name / "name"
                ).read_text()
            except OSError:
                continue

            if not any(name in adapter.casefold() for name in cls.NON_DISPLAY_ADAPTERS):
                buses.append(i2c_dev)

        return buses

    async def _get_display(self) -> BacklightDevice | None:
        await self.close()

        drm_cards = await self.detect_connected_drm()

        ret = None

        disconnected = []

        i2c_re = re.compile(r"^i2c-(\d+)$")

        for i2c_dev in self._candidate_buses(drm_cards):
            i2c_num = -1

            mul = 0.0
            max_level = 0

            name, serial, card_name = None, None, None

            try:
                self._open_sync(i2c_dev, self.I2C_ADDR_EDID)
                name, serial = await self.get_edid()  # This takes forever
            except (OSError, ValueError):
                pass
            finally:
                await self.close()

            if name is not None or serial is not None:
                try:
                    self._open_sync(i2c_dev, self.I2C_ADDR_TX)
                    max_level = (await self.read(self.I2C_IDX_BL))[0]
                    mul = max_level / 100
                except (OSError, ValueError, IndexError):
                    pass
                finally:
                    await self.close()

                if (match := i2c_re.fullmatch(i2c_dev.name)) is not None:
                    i2c_num = int(match.group(1))

                if i2c_num >= 0 and mul > 0:
                    card_name = self.i2c_to_drm(i2c_num)

                    is_connected = False

                    for _card_name, _i2c_num, connected in drm_cards:
                        if _card_name == card_name or _i2c_num == i2c_num:
                            is_connected = connected
                            break

                    dev = BacklightDevice(
                        name,
                        serial,
                        i2c_dev,
                        i2c_num,
                        card_name,
                        is_connected,
                        mul,
                        max_level,
                    )

                    if self.disp is None or dev.matches(self.disp):
                        if not dev.connected:
                            disconnected.append(dev)
                        else:
                            ret = dev
                            break

            await anyio.lowlevel.checkpoint()

        if ret is None and disconnected:
            ret = disconnected[0]

        return ret

    async def init(self):
        device = await self._get_display()
        await self.close()

        if device is None:
            raise OSError(errno.ENODEV, f"{self.disp}: no compatible I2C device found")
        else:
            self.device = device

        if self.fd < 0:
            self._open_sync(self.device.path, self.I2C_ADDR_TX)

    def _open_sync(self, path: str | Path, addr: int) -> int:

        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)

        try:
            # EBUSY if a kernel driver owns the address, e.g. an EEPROM driver
            # on a memory bus: never take it over
            fcntl.ioctl(self.fd, self.I2C_SLAVE, addr)
        except OSError as ex:
            os.close(self.fd)
            self.fd = -1

            if ex.errno == errno.EBUSY and addr == self.I2C_ADDR_TX:
                logging.info(
                    "%s: DDC/CI is used by a kernel driver, e.g. ddcci_backlight, "
                    "use the 'acpi' driver for its backlight device",
                    path,
                )
            raise

        return self.fd

    async def _read(self, n: int) -> bytes:

        buff = await self.fd_read_async(n + 3)
        # buff = os.read(self.fd, n + 3)

        if buff[0] != self.I2C_SRC_ADDR:
            raise ValueError("response from unknown source")

        if functools.reduce(operator.xor, buff) != self.I2C_CKSUM_XOR:
            raise ValueError("checksum verification failed")

        if len(buff) < ((buff[1] & ~self.I2C_WR_LEN) + 3):
            raise ValueError("unexpected response length")

        return buff[2:-1]

    async def _write(self, *data: int) -> int:
        msg = bytearray(data)
        msg.insert(0, len(msg) | self.I2C_WR_LEN)
        msg.insert(0, self.I2C_WR_SUB)
        msg.append(functools.reduce(operator.xor, msg, self.I2C_SRC_ADDR))

        return await self.fd_write_async(msg)

    async def write(self, vcpopcode: int, value: int) -> int:
        return await self._write(self.I2C_CHG, vcpopcode, *value.to_bytes(2, "big"))

    async def read(self, vcpopcode: int) -> tuple[int, int]:
        await self._write(self.I2C_READ, vcpopcode)
        await anyio.sleep(self.API_PAUSE)
        buff = await self._read(self.BUFFSZ_READ)

        if buff[0] != self.I2C_PRE_VALUE:
            raise ValueError("not a feature response")

        if buff[1] != 0:
            raise ValueError("VCP opcode not supported")

        if buff[2] != vcpopcode:
            raise ValueError("unexpected response")

        # NOTE: buff[3] is a result code
        return int.from_bytes(buff[4:6], "big"), int.from_bytes(buff[6:8], "big")

    async def get_edid(self) -> tuple[str | None, str | None]:
        await self.fd_write_async(self.I2C_IDX_EDID.to_bytes())
        await anyio.sleep(self.API_PAUSE)
        # buff = os.read(self.fd, self.BUFFSZ_EDID)
        buff = await self.fd_read_async(self.BUFFSZ_EDID)
        return self.get_dev_id(buff)

    async def get_level(self) -> float:
        if self.device is None:
            return 0.0
        return (await self.read(self.I2C_IDX_BL))[1] / self.device.mul

    async def set_level(self, value: int | float):
        if self.device is not None:
            raw = int(max(0, min(value * self.device.mul, self.device.max_level)))
            await self.write(self.I2C_IDX_BL, raw)
