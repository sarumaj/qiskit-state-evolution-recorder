import pytest

from qiskit_state_evolution_recorder.compatibility import proxy_obj


class SingleMatchObject:
    """Test class with various attribute types for testing the proxy."""

    def __init__(self):
        self.public_attr = "public_value"
        self._private_attr = "private_value"
        self.__name_attr = "name_value"
        self._SingleMatchObject__name_attr = "mangled_value"

    def public_method(self):
        return "public_method"

    def _private_method(self):
        return "private_method"

    def method_with_args(self, arg1, arg2):
        return f"method_with_args: {arg1}, {arg2}"

    def method_with_kwargs(self, **kwargs):
        return f"method_with_kwargs: {kwargs}"


class SingleMatchClass:
    """Test class for testing proxy with class objects."""

    class_attr = "class_value"

    def __init__(self):
        self.instance_attr = "instance_value"

    @classmethod
    def class_method(cls):
        return "class_method"


class MultipleMatchObject:
    """Test class that will trigger multiple attribute matches."""

    def __init__(self):
        self.attr = "public_attr"
        self._attr = "private_attr"


def test_proxy_public_attribute():
    """Test accessing public attributes through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # The proxy wraps the attribute in another proxy
    assert proxy.public_attr.orig_obj == "public_value"


def test_proxy_private_attribute():
    """Test accessing private attributes through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Should access _private_attr and wrap it in a proxy
    assert proxy.private_attr.orig_obj == "private_value"


def test_proxy_name_mangled_attribute():
    """Test accessing name-mangled attributes through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # The proxy finds _SingleMatchObject__name_attr (the mangled version) first
    # because it's explicitly defined in the class
    assert proxy.name_attr.orig_obj == "mangled_value"


def test_proxy_mangled_attribute():
    """Test accessing explicitly mangled attributes through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Should access _SingleMatchObject__name_attr
    assert proxy._SingleMatchObject__name_attr.orig_obj == "mangled_value"


def test_proxy_multiple_matches():
    """Test that proxy raises error when multiple attributes match."""
    obj = MultipleMatchObject()
    proxy = proxy_obj(obj)

    # Both attr and _attr exist, so this should raise an error
    with pytest.raises(AttributeError, match="MultipleMatchObject object has multiple attributes attr"):
        _ = proxy.attr


def test_proxy_nonexistent_attribute():
    """Test that proxy raises error for nonexistent attributes."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    with pytest.raises(AttributeError, match="SingleMatchObject object has no attribute nonexistent"):
        _ = proxy.nonexistent


def test_proxy_original_object_access():
    """Test accessing the original object through proxy_obj property."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    assert proxy.orig_obj is obj
    assert proxy.orig_obj.public_attr == "public_value"


def test_proxy_with_class():
    """Test proxy with a class object."""
    proxy = proxy_obj(SingleMatchClass)

    assert proxy.class_attr.orig_obj == "class_value"
    assert proxy.__name__.orig_obj == "SingleMatchClass"


def test_proxy_with_instance():
    """Test proxy with a class instance."""
    obj = SingleMatchClass()
    proxy = proxy_obj(obj)

    assert proxy.instance_attr.orig_obj == "instance_value"
    assert proxy.class_attr.orig_obj == "class_value"


def test_proxy_nested_access():
    """Test accessing nested attributes through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Test that we can access the original object and then access its attributes
    original = proxy.orig_obj
    assert original.public_attr == "public_value"


def test_proxy_method_access():
    """Test accessing methods through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Methods are wrapped in proxy objects
    method_proxy = proxy.public_method
    assert hasattr(method_proxy, "orig_obj")
    # The actual method is accessible through proxy_obj
    assert method_proxy.orig_obj() == "public_method"


def test_proxy_private_method_access():
    """Test accessing private methods through proxy."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Private methods are wrapped in proxy objects
    method_proxy = proxy._private_method
    assert hasattr(method_proxy, "orig_obj")
    # The actual method is accessible through proxy_obj
    assert method_proxy.orig_obj() == "private_method"


def test_proxy_method_call_via_original():
    """Test calling methods by accessing the original object."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Access the original object to call methods directly
    assert proxy.orig_obj.public_method() == "public_method"
    assert proxy.orig_obj._private_method() == "private_method"


def test_proxy_method_with_args_via_original():
    """Test calling methods with arguments via original object."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Test method with positional arguments via original object
    assert proxy.orig_obj.method_with_args("arg1", "arg2") == "method_with_args: arg1, arg2"


def test_proxy_method_with_kwargs_via_original():
    """Test calling methods with keyword arguments via original object."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Test method with keyword arguments via original object
    result = proxy.orig_obj.method_with_kwargs(key1="value1", key2="value2")
    assert "key1" in result and "value1" in result
    assert "key2" in result and "value2" in result


def test_proxy_class_method_via_original():
    """Test calling class methods via original object."""
    proxy = proxy_obj(SingleMatchClass)

    # Class methods can be called via the original object
    assert proxy.orig_obj.class_method() == "class_method"


def test_proxy_with_builtin_object():
    """Test proxy with a built-in object like a string."""
    string_obj = "test_string"
    proxy = proxy_obj(string_obj)

    # Access built-in methods via the original object
    assert proxy.orig_obj.upper() == "TEST_STRING"
    assert proxy.orig_obj.lower() == "test_string"
    assert len(proxy.orig_obj) == 11


def test_proxy_with_dict():
    """Test proxy with a dictionary object."""
    dict_obj = {"key1": "value1", "key2": "value2"}
    proxy = proxy_obj(dict_obj)

    # Access dict methods via the original object
    assert "key1" in proxy.orig_obj
    assert proxy.orig_obj.get("key1") == "value1"
    assert len(proxy.orig_obj) == 2


def test_proxy_attribute_chaining():
    """Test chaining attribute access through multiple proxies."""
    obj = SingleMatchObject()
    proxy = proxy_obj(obj)

    # Access nested attributes through the proxy chain
    # proxy.public_attr returns a proxy, then .orig_obj gets the actual value
    assert proxy.public_attr.orig_obj == "public_value"

    # Can also access the original object and then access attributes normally
    assert proxy.orig_obj.public_attr == "public_value"
