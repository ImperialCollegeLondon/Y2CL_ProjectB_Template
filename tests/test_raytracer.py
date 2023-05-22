from inspect import signature
import pytest
from fixtures import *

class TestOpticalElement:
    def test_oe_exists(self, elements):
        assert hasattr(elements, "OpticalElement")

    def test_pr_exists(self, elements):
        assert hasattr(elements.OpticalElement, "propagate_ray")

    def test_pr_raises(self, elements):
        with pytest.raises(NotImplementedError):
            elements.OpticalElement.propagate_ray(1, 2)


class TestSphericalRefraction:
    def test_sr_exists(self, elements):
        assert hasattr(elements, "SphericalRefraction")

    def test_inheritance(self, elements):
        assert elements.OpticalElement in elements.SphericalRefraction.mro()

    def test_sr_intercept_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "intercept")

    def test_sr_pr_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "propagate_ray")

    def test_num_parameters(self, elements):
        assert len(signature(elements.SphericalRefraction).parameters) == 5

    def test_alex(self, rt):
        # test combine environment
        assert "Ray" in rt
        assert "OpticalElement" in rt
        r = rt.Ray()
        assert r.direc().dtype == "float"

class TestOutputPlane:
    def test_op_exists(self, elements):
        assert hasattr(elements, "OutputPlane")

    def test_inheritance(self, elements):
        assert elements.OpticalElement in elements.OutputPlane.mro()

    def test_intercept_exists(self, elements):
        assert hasattr(elements.OutputPlane, "intercept")

    def test_pr_exists(self, elements):
        assert hasattr(elements.OutputPlane, "propagate_ray")

    def test_num_parameters(self, elements):
        assert len(signature(elements.OutputPlane).parameters) == 2

class TestAdvancedDesign:
    def test_ray_bundle_exists(self, ray_bundle):
        assert ray_bundle is not None

    def test_ray_bundle_class(self, ray_bundle):
        assert isinstance(ray_bundle, type)

    def test_bundle_not_inheritance(self, rays, ray_bundle):
        assert rays.Ray not in ray_bundle.mro()

    def test_bundle_rms(self, ray_bundle):
        present = hasattr(ray_bundle, "rms") or \
            hasattr(ray_bundle, "RMS") or \
            hasattr(ray_bundle, "spot_radius")
        assert present

    def test_sr_tree(self, elements):
        test_tree = set(elements.SphericalRefraction.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.SphericalRefraction,
                                     elements.OpticalElement,
                                     object})

    def test_intercept_not_in_sr(self, elements):
        assert "intercept" not in vars(elements.SphericalRefraction)

    def test_op_tree(self, elements):
        test_tree = set(elements.OutputPlane.mro())
        assert len(test_tree) > 3
        assert test_tree.difference({elements.OutputPlane,
                                     elements.OpticalElement,
                                     object})
        
    def test_intercept_not_in_op(self, elements):
        assert "intercept" not in vars(elements.OutputPlane)
