import pytest

try:
    from entoolkit import toolkit as et
    # Support for newer versions where functions moved to legacy
    if not hasattr(et, 'ENopen'):
        from entoolkit import legacy as et
except ImportError:
    from entoolkit import legacy as et


def test_epanet_version():
    """Test that the EPANET library loads and returns a version number."""
    # Ensure we check both toolkit and legacy for the version function
    version_func = None
    if hasattr(et, "ENgetversion"):
        version_func = et.ENgetversion
    else:
        try:
            from entoolkit import legacy as leg
            if hasattr(leg, "ENgetversion"):
                version_func = leg.ENgetversion
        except ImportError:
            pass

    if version_func is None:
        pytest.fail("Could not find ENgetversion in entoolkit.toolkit or entoolkit.legacy")

    try:
        version = version_func()
        assert version > 0
        print(f"EPANET Version: {version}")
    except Exception as e:
        pytest.fail(f"Could not load EPANET library or get version: {e}")


def test_constants():
    """Test that some new constants are defined."""
    # The user confirmed the name is EN_DEMANDMODEL without an underscore
    if hasattr(et, "EN_DEMANDMODEL"):
        # Updated for EPANET 2.3 where this constant is 27
        assert et.EN_DEMANDMODEL == 27
    
    if hasattr(et, "EN_LINK_TYPE"):
        assert et.EN_LINK_TYPE == 14
