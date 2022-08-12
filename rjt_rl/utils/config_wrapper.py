import importlib
import logging
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

logger = logging.getLogger(__name__)

ConfigContainer = TypeVar("ConfigContainer", bound=BaseContainer)


def load_class(class_name: str, default_module_name: Optional[str] = None) -> Type:
    if "." not in class_name:
        if default_module_name is not None:
            class_name = default_module_name + "." + class_name
        else:
            raise ValueError(f"cannot load class from empty module path: {class_name}")
    name_list = class_name.split(".")
    modnm = ".".join(name_list[:-1])
    clsnm = name_list[-1]
    logger.info(f"loading {clsnm} from {modnm}")
    m = importlib.import_module(modnm)
    cls: Type = getattr(m, clsnm)
    return cls


def _validate_config_impl(cls: Type, user_config: ConfigContainer) -> ConfigContainer:
    cfg_cls = cls.get_config_class()
    schema = OmegaConf.structured(cfg_cls)
    typed_config = OmegaConf.merge(schema, user_config)
    return cast(ConfigContainer, typed_config)


def validate_config_by_class(
    cls: Type, user_config: ConfigContainer
) -> ConfigContainer:
    if not hasattr(cls, "get_config_class"):
        logger.warning(
            f"cls {cls} does not have get_config_class method. "
            "no type check is performed."
        )
        return user_config
    return _validate_config_impl(cls, user_config)


def obj_from_config(
    class_factory: Callable[[str], Any], config: ConfigContainer, *args: Any
) -> Tuple[Any, Optional[ConfigContainer]]:
    assert "class" in config
    cls_name = config["class"]
    cls = class_factory(cls_name)
    if hasattr(cls, "get_config_class"):
        copied_config = cast(DictConfig, config).copy()
        copied_config.pop("class")
        typed_config = _validate_config_impl(cls, copied_config)
        result_config = OmegaConf.merge({"class": cls_name}, typed_config)
        return cls.from_config(typed_config, *args), cast(
            ConfigContainer, result_config
        )
    else:
        logger.warning(f"class {cls} does not have get_config_class method.")
        return cls(*args), None


def _merge_configs(
    yaml_files: Sequence[str],
    ovwr_config: Optional[BaseContainer] = None,
) -> BaseContainer:
    user_config: BaseContainer = OmegaConf.create()
    for fname in yaml_files:
        conf = OmegaConf.load(fname)
        user_config = OmegaConf.merge(user_config, conf)
    if ovwr_config is not None:
        user_config = OmegaConf.merge(user_config, ovwr_config)
    return user_config
