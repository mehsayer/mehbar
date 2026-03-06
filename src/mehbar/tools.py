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

        for literal, fld, _, _ in self.parse(format_string):
            if fld is None or kwargs.get(fld) is None:
                unparsed += literal
            elif fld in kwargs and str(kwargs[fld]):
                unparsed += literal + "{" + fld + "}"

        return super().vformat(unparsed, args, kwargs)
