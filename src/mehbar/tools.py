from typing import Sequence, Mapping, Any
import string

def overlay_dict_r(bottom: dict[Any, Any], top: dict[Any, Any], max_depth: int = 10, depth: int = 0):
    if depth > max_depth:
        raise ValueError(f"maximum nesting depth exceeded: {max_depth}")

    for ktop, vtop in top.items():
        if isinstance(vtop, dict):
            if ktop not in bottom or not isinstance(bottom[ktop], dict):
                bottom[ktop] = {}
            overlay_dict_r(bottom[ktop], vtop, max_depth, depth + 1)
        else:
            bottom[ktop] = vtop


class FormattableTimeDelta:

    def __init__(self, seconds: float | int):
        self.tot_secs = int(seconds)
        self.tot_mins, self.secs = divmod(self.tot_secs, 60)
        self.tot_hrs, self.mins = divmod(self.tot_mins, 60)
        self.days, self.hrs = divmod(self.tot_hrs, 24)


    def strftime(self, format_str: str) -> str:
        fmt = []
        push = fmt.append

        i, n = 0, len(format_str)

        while i < n:
            ch = format_str[i]
            i += 1
            if ch == '%':
                if i < n:
                    ch = format_str[i]
                    i += 1
                    match ch:
                        case 's':
                            push('%d' % self.secs)
                        case 'S':
                            push('%02d' % self.secs)
                        case 'a':
                            push('%d' % self.tot_secs)
                        case 'A':
                            push('%02d' % self.tot_secs)
                        case 'm':
                            push('%d' % self.mins)
                        case 'M':
                            push('%02d' % self.mins)
                        case 'b':
                            push('%d' % self.tot_mins)
                        case 'B':
                            push('%02d' % self.tot_mins)
                        case 'h':
                            push('%d' % self.hrs)
                        case 'H':
                            push('%02d' % self.hrs)
                        case 'i':
                            push('%d' % self.tot_hrs)
                        case 'I':
                            push('%02d' % self.tot_hrs)
                        case 'd':
                            push('%d' % self.days)
                        case 'D':
                            push('%02d' % self.days)
                        case _:
                            push('%')
                            push(ch)
                else:
                    push('%')
            else:
                push(ch)

        return ''.join(fmt)

    def __format__(self, format_str: str) -> str:
        return self.strftime(format_str) if format_str else self.__str__()


    def __str__(self) -> str:
        return self.strftime("%I:%M:%S")

    def __int__(self) -> int:
        return self.tot_secs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(seconds={self.tot_secs})'


class OptionalFormatter(string.Formatter):
    """Like the default stripng formatter that you know and love but silently
    skips missing fields.
    """

    def get_fields(self, format_string) -> list[str]:
        ret = []
        for _, fld, _, _ in self.parse(format_string):
            if fld is not None:
                ret.append(fld)

        return ret

    def vformat(
        self,
        format_string: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str:

        if args:
            raise ValueError("non-keyword arguments are not supported")

        unparsed = str()

        for literal, fld, spec, _ in self.parse(format_string):
            if fld is None or kwargs.get(fld) is None:
                unparsed += literal
            elif fld in kwargs and str(kwargs[fld]):
                unparsed += literal + "{" + fld
                if spec is not None:
                    unparsed += ':' + spec
                unparsed += "}"

        return super().vformat(unparsed, args, kwargs)
