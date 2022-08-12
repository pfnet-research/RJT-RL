from os import PathLike
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    PathLikeType = PathLike[Any]
else:
    PathLikeType = PathLike
AnyPathLikeType = Union[str, PurePath, PathLikeType]
