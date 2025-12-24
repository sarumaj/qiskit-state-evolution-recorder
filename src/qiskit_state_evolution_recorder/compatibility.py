from typing import Any


def proxy_obj(obj: Any) -> Any:
    """
    Workaround to ensure compatibility with qiskit>=2.0.0
    It allows to access public and private properties of an object
    by specifying the name of the property.

    Parameters:
    -----------
    obj: Any
        The object to proxy

    Returns:
    --------
    Any
        The object wrapped in a Proxy
    """

    class Proxy:
        def __init__(self, obj: Any):
            """Initialize the proxy.

            Parameters:
            -----------
            obj: Any
                The object to proxy
            """
            self._obj = obj
            self._name = obj.__name__ if isinstance(obj, type) else obj.__class__.__name__

        @property
        def orig_obj(self) -> Any:
            """Get the original object.

            Returns:
            --------
            Any
                The original object
            """
            return self._obj

        def __getattr__(self, name: str) -> Any:
            """Get an attribute of the proxy.

            Parameters:
            -----------
            name: str
                The name of the attribute

            Returns:
            --------
            Any
                The attribute value wrapped in an Proxy
            """
            props = (
                name,
                f"_{name}",
                f"__{name}",
                f"_{self._name}__{name}",
            )
            if name.startswith("_"):
                props = (
                    name,
                    f"_{self._name}__{name}",
                    name[1:],
                )
            results = [(prop, Proxy(getattr(self._obj, prop))) for prop in props if hasattr(self._obj, prop)]
            if len(results) == 1:
                return results[0][1]
            elif len(results) > 1:
                raise AttributeError(f"{self._name} object has multiple attributes {name}: {[p for p, _ in results]}")
            raise AttributeError(f"{self._name} object has no attribute {name}")

    return Proxy(obj)
