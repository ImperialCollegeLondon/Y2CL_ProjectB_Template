from fixtures import *
from inspect import signature
from types import FunctionType
from unittest.mock import MagicMock
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal
import pytest
from utils import check_figures_equal

class TestTask1:
    def test_docstring_present(self, ph):
        assert ph.__doc__ != None
    def test_docstring_not_blank(self, ph):
        assert ph.__doc__ != ""

class TestTask2:
    def test_p_exists(self, rays):
        assert hasattr(rays.Ray, "pos")

    def test_pos_method(self, rays):
        assert isinstance(rays.Ray.pos, FunctionType)

    def test_default_p_array(self, rays):
        r = rays.Ray()
        assert isinstance(r.pos(), np.ndarray)

    def test_p_array(self, test_ray):
        assert isinstance(test_ray.pos(), np.ndarray)

    def test_p_set(self, test_ray):
        assert np.allclose(test_ray.pos(), [1., 2., 3.])

    def test_k_exists(self, rays):
        assert hasattr(rays.Ray, "direc")

    def test_direc_method(self, rays):
        assert isinstance(rays.Ray.direc, FunctionType)

    def test_k_array(self, test_ray):
        assert isinstance(test_ray.direc(), np.ndarray)

    def test_k_set(self, test_ray):
        test_k = np.array([4.,5.,6.])
        test_k /= np.linalg.norm(test_k)
        assert np.allclose(test_ray.direc(), test_k)

    def test_append_exists(self, rays):
        assert hasattr(rays.Ray, "append")

    def test_append_method(self, rays):
        assert isinstance(rays.Ray.append, FunctionType)

    def test_append_increases_length(self, default_ray, var_name_map):
        pos = getattr(default_ray, var_name_map['pos'])
        assert len(pos) == 1
        default_ray.append([1.,2.,3.], [4.,5.,6.])
        assert len(pos) == 2

    def test_append_pos_type(self, default_ray, var_name_map):
        assert isinstance(getattr(default_ray,var_name_map["pos"]), list)
        default_ray.append(pos=[4.,5.,6.], direc=[7.,8.,9.])
        assert isinstance(getattr(default_ray,var_name_map["pos"]), list)

    def test_append_pos_array(self, default_ray, var_name_map):
        default_ray.append(pos=[4.,5.,6.], direc=[7.,8.,9.])
        pos = getattr(default_ray,var_name_map["pos"])
        assert isinstance(pos[-1], np.ndarray)

    def test_append_pos(self, default_ray, var_name_map):
        default_ray.append(pos=[1., 2., 3.], direc=[4., 5., 6.])
        pos = getattr(default_ray, var_name_map['pos'])
        assert np.allclose(pos[1], [1., 2., 3.])
        assert np.allclose(default_ray.pos(), [1., 2., 3.])

    def test_append_direc_array(self, default_ray, var_name_map):
        default_ray.append(pos=[4.,5.,6.], direc=[7.,8.,9.])
        direc = getattr(default_ray, var_name_map['direc'])
        assert isinstance(direc, np.ndarray)

    def test_append_direc(self, default_ray, var_name_map):
        default_ray.append([1., 2., 3.], [4., 5., 6.])
        direc = getattr(default_ray, var_name_map["direc"])
        test_k = np.array([4.,5.,6.])
        test_k /= np.linalg.norm(test_k)
        assert np.allclose(direc, test_k)
        assert np.allclose(default_ray.direc(), test_k)

    def test_append_pos_iterable_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=12, direc=[1.,2.,3.])

    def test_append_pos_too_long_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1., 2., 3., 4.], direc=[1.,2.,3.])

    def test_append_pos_too_short_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1., 2.], direc=[1.,2.,3.])

    def test_append_direc_iterable_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1.,2.,3.], direc=12)

    def test_append_direc_too_long_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1.,2.,3.], direc=[1., 2., 3., 4.])

    def test_append_direc_too_short_check(self, default_ray):
        with pytest.raises(Exception):
            default_ray.append(pos=[1.,2.,3.], direc=[1., 2.])

    def test_vertices_exists(self, rays):
        assert hasattr(rays.Ray, "vertices")

    def test_vertices_method(self, rays):
        assert isinstance(rays.Ray.vertices, FunctionType)

    def test_vertices_list(self, default_ray):
        assert isinstance(default_ray.vertices(), list)

    def test_vertices_array(self, default_ray):
        default_ray.append([4.,5.,6.], [7.,8.,9.])
        for vertex in default_ray.vertices():
            assert isinstance(vertex, np.ndarray)

    def test_vertices(self, test_ray):
        assert np.allclose(test_ray.vertices(), [[1.,2.,3.]])

    def test_multiple_vertices(self, test_ray):
        test_ray.append([4.,5.,6.], [7.,8.,9.])
        assert np.allclose(test_ray.vertices(),
                           [[1.,2.,3.],
                            [4.,5.,6.]])


class TestTask3:
    def test_sr_class_exists(self, elements):
        assert hasattr(elements, "SphericalRefraction")
    def test_inheritance(self, elements):
        assert elements.OpticalElement in elements.SphericalRefraction.mro()
    def test_num_parameters(self, elements):
        assert len(signature(elements.SphericalRefraction).parameters) == 5

class TestTask4:
    def test_intercept_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "intercept")

    def test_num_parameters(self, elements):
        assert len(signature(elements.SphericalRefraction.intercept).parameters) == 2

    def test_no_intercept(self, rays, elements):
        ray = rays.Ray(pos=[10., 0., 0.])
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=5.)
        assert sr.intercept(ray) is None

    def test_intercept_array(self, default_ray, elements):
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        assert isinstance(sr.intercept(default_ray), np.ndarray)

    def test_onaxis_ray_convex_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0.,0.,0.])
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray), [0., 0., 10.])

    def test_onaxis_ray_concave_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0.,0.,0.])
        sr = elements.SphericalRefraction(z0=10, curvature=-0.02, n1=1., n2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray), [0., 0., 10.])

    def test_offaxis_convex_intercept(self, rays, elements):
        ray1 = rays.Ray(pos=[1., 0., 0.])
        ray2 = rays.Ray(pos=[-1., 0., 0.])
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [1., 0., 10.010001])
        assert np.allclose(sr.intercept(ray2), [-1., 0., 10.010001])

    def test_offaxis_concave_intercept(self, rays, elements):
        ray1 = rays.Ray(pos=[1., 0., 0.])
        ray2 = rays.Ray(pos=[-1., 0., 0.])
        sr = elements.SphericalRefraction(z0=10, curvature=-0.02, n1=1., n2=1.5, aperture=50.)
        assert np.allclose(sr.intercept(ray1), [1., 0., 9.989999])
        assert np.allclose(sr.intercept(ray2), [-1., 0., 9.989999])

    def test_ray_pos_called(self, rays, elements, monkeypatch):
        pos_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        monkeypatch.setattr(rays.Ray, "pos", pos_mock)
        r = rays.Ray()
        sr = elements.SphericalRefraction(z0=10, curvature=-0.02, n1=1., n2=1.5, aperture=50.)
        sr.intercept(r)
        pos_mock.assert_called_once()

    def test_ray_direc_called(self, rays, elements, monkeypatch):
        direc_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        monkeypatch.setattr(rays.Ray, "direc", direc_mock)
        r = rays.Ray()
        sr = elements.SphericalRefraction(z0=10, curvature=-0.02, n1=1., n2=1.5, aperture=50.)
        sr.intercept(r)
        direc_mock.assert_called_once()


class TestTask5:
    def test_refract_exists(self, ph):
        assert hasattr(ph, "refract")

    def test_num_params(self, ph):
        assert len(signature(ph.refract).parameters) == 4

    def test_returns_unitvector(self, ph):
        dir = np.array([0., 0.05, 1.])
        norm = np.array([0., -1.9, -1.])
        assert np.linalg.norm(ph.refract(dir, norm, 1.0, 1.5)) == 1.

    def test_onaxis_refract(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 0., -1.])
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.5), np.array([0., 0., 1.]))

    def test_offaxis_refract_lower(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., -1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.5), np.array([0., 0.29027623, 0.9569429]))

    def test_offaxis_refract_upper(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.5), np.array([0., -0.29027623, 0.9569429]))

    def test_equal_ref_indices_onaxis(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 0., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.0), np.array([0., 0., 1.]))
    
    def test_equal_ref_indices_upper(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.0), np.array([0., 0., 1.]))

    def test_equal_ref_indices_lower(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., -1., -1.])
        norm /= np.linalg.norm(norm)
        assert np.allclose(ph.refract(dir, norm, 1.0, 1.0), np.array([0., 0., 1.]))

    def test_TIR(self, ph):
        dir = np.array([0., 0., 1.])
        norm = np.array([0., 1., -1.])
        norm /= np.linalg.norm(norm)
        assert ph.refract(dir, norm, 1.5, 1.0) is None

class TestTask6:
    def test_pr_exists(self, elements):
        assert hasattr(elements.SphericalRefraction, "propagate_ray")

    def test_num_parameters(self, elements):
        assert len(signature(elements.SphericalRefraction.propagate_ray).parameters) == 2

    def test_pr_calls_intercept(self, elements, monkeypatch):
        intercept_mock = MagicMock(return_value=np.array([0., 0., 10.]))
        monkeypatch.setattr(elements.SphericalRefraction, "intercept", intercept_mock)
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        ray = elements.Ray([0.,0.,0.], [0.,0.,1.])
        sr.propagate_ray(ray)
        intercept_mock.assert_called_once_with(ray)

    def test_pr_calls_refract(self, rays, elements, ph, monkeypatch):
        refract_mock = MagicMock(return_value=np.array([0., 0., 1.]))
        monkeypatch.setattr(ph, "refract", refract_mock)
        if hasattr(elements, "refract"):
            monkeypatch.setattr(elements, "refract", refract_mock)
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        ray = rays.Ray([0.,0.,0.], [0.,0.,1.])
        sr.propagate_ray(ray)
        refract_mock.assert_called_once()
        k, norm, n1, n2 = refract_mock.call_args.args
        norm /= np.linalg.norm(norm)
        assert np.allclose(k, ray.direc())
        assert np.allclose(norm, np.array([0., 0., -1.]))
        assert n1 == 1.0
        assert n2 == 1.5

    def test_pr_calls_append(self, rays, elements, monkeypatch):
        append_mock = MagicMock(return_value=None)
        monkeypatch.setattr(rays.Ray, "append", append_mock)
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        sr.propagate_ray(rays.Ray([1., 2., 0.], [0., 0., 1.]))
        append_mock.assert_called_once()
        new_pos, new_direc = append_mock.call_args.args
        assert np.allclose(new_pos, [1., 2., 10.05002503])
        assert np.allclose(new_direc, [-0.00667112, -0.01334223, 0.99988873])

class TestTask7:
    def test_exists(self, an):
        assert hasattr(an, "task7")

    def test_doesnt_crash(self, an):
        an.task7()

    def test_ray_called(self, an, rays, monkeypatch):
        mock_ray_init = MagicMock(wraps=rays.Ray)
        monkeypatch.setattr(rays, "Ray", mock_ray_init)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an, "Ray", mock_ray_init)
        try:
            an.task7()
        except: pass
        mock_ray_init.assert_called()

    def test_sr_called(self, an, elements, monkeypatch):
        mock_sr_class = MagicMock()
        monkeypatch.setattr(elements, "SphericalRefraction", mock_sr_class)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an, "SphericalRefraction", mock_sr_class)
        try:
            an.task7()
        except: pass
        mock_sr_class.assert_called()

    def test_sr_calls_propagate(self, an, elements, monkeypatch):
        mock_pr = MagicMock()
        monkeypatch.setattr(elements.SphericalRefraction, "propagate_ray", mock_pr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an.SphericalRefraction, "propagate_ray", mock_pr)
        try:
            an.task7()
        except: pass
        mock_pr.assert_called()

    def test_multiple_rays_created(self, an, rays, monkeypatch):
        mock_ray_init = MagicMock(wraps=rays.Ray)
        monkeypatch.setattr(rays, "Ray", mock_ray_init)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an, "Ray", mock_ray_init)
        try:
            an.task7()
        except: pass
        assert mock_ray_init.call_count > 1

    def test_multiple_propagates(self, an, elements, monkeypatch):
        mock_pr = MagicMock()
        monkeypatch.setattr(elements.SphericalRefraction, "propagate_ray", mock_pr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an.SphericalRefraction, "propagate_ray", mock_pr)
        try:
            an.task7()
        except: pass
        assert mock_pr.call_count > 1

    @pytest.mark.skip(reason="dont remember what this was testing.")
    def test_plot(self, rays, elements):
        ray = rays.Ray([0., 0., 0.], [0., 0., 1.])
        sr = elements.SphericalRefraction(z0=10, curvature=0.02, n1=1., n2=1.5, aperture=50.)
        sr.propagate_ray(ray)

    @pytest.mark.skip(reason="Unfinished idea.")
    def test_get_data(self):
        x = np.arange(1,10,0.2)
        y=x**2
        import matplotlib.pyplot as plt
        import scipy.stats as sps
        fig = plt.figure()
        plt.plot(x, y, 'r--')
        a = fig.gca().lines[0].get_xydata()[:,0]
        b = fig.gca().lines[0].get_ydata()
        print(a)
        print(b)
        rand = sps.norm.rvs(loc=1, scale=0.2, size=len(x))
        print(sps.ks_2samp(b, x**2 * rand).pvalue)
        plt.plot(x, x**2 * rand, 'b--')
        plt.show()
        assert 1 == a

class TestTask8:
    def test_op_exists(self, elements):
        assert hasattr(elements, "OutputPlane")

    def test_intercept_exists(self, elements):
        assert hasattr(elements.OutputPlane, "intercept")

    def test_propagate_exists(self, elements):
        assert hasattr(elements.OutputPlane, "propagate_ray")

    def test_pr_calls_intercept(self, default_ray, elements, monkeypatch):
        intercept_patch = MagicMock(return_value=np.array([1., 2., 3.]))
        monkeypatch.setattr(elements.OutputPlane, "intercept", intercept_patch)
        op = elements.OutputPlane(10)
        op.propagate_ray(default_ray)
        intercept_patch.assert_called_once_with(default_ray)

    def test_pr_doesnt_call_refract(self, default_ray, ph, elements, monkeypatch):
        refract_mock = MagicMock(return_value=np.array([0., 0., 1.]))
        # mock both places incase they have imported the functing into elements
        monkeypatch.setattr(ph, "refract", refract_mock)
        if hasattr(elements, "refract"):
            monkeypatch.setattr(elements, "refract", refract_mock)
        op = elements.OutputPlane(z0=10)
        op.propagate_ray(default_ray)
        refract_mock.assert_not_called()
       
    def test_pr_calls_append(self, default_ray, rays, elements, monkeypatch):
        append_mock = MagicMock(return_value=None)
        monkeypatch.setattr(rays.Ray, "append", append_mock)
        op = elements.OutputPlane(z0=10)
        intercept = op.intercept(default_ray)
        op.propagate_ray(default_ray)
        append_mock.assert_called_once()
        new_pos, _ = append_mock.call_args.args
        assert np.allclose(new_pos, intercept)

    def test_parallel_intercept(self, rays, elements):
        ray = rays.Ray(pos=[0., 10., 0.])
        op = elements.OutputPlane(z0=10)
        intercept = op.intercept(ray)
        assert np.allclose(intercept, [0., 10., 10.])

    def test_nonparallel_intercept(self, rays, elements):
        ray = rays.Ray(pos=[1., 10., 2.], direc=[-0.15, -0.5, 1.])
        op = elements.OutputPlane(z0=10)
        intercept = op.intercept(ray)
        assert np.allclose(intercept, [-0.2, 6., 10.])


class TestTask9:

    def test_task9_exists(self, an):
        assert hasattr(an, "task9")

    def test_sr_called(self, an, elements, monkeypatch):
        sr = MagicMock(wraps=elements.SphericalRefraction)
        monkeypatch.setattr(elements, "SphericalRefraction", sr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an, "SphericalRefraction", sr)
        an.task9()
        sr.assert_called_once()

    def test_sr_setup(self, an, elements, monkeypatch):
        sr = MagicMock(wraps=elements.SphericalRefraction)
        monkeypatch.setattr(elements, "SphericalRefraction", sr)
        if hasattr(an, "SphericalRefraction"):
            monkeypatch.setattr(an, "SphericalRefraction", sr)
        an.task9()
        sr.assert_called_once()
        args = set(sr.call_args.args)
        args.update(sr.call_args.kwargs.values())
        assert len(args) == 5
        assert not {100, 0.03, 1., 1.5}.difference(args)

    def test_op_called(self, an, elements, monkeypatch):
        op = MagicMock(wraps=elements.OutputPlane)
        monkeypatch.setattr(elements, "OutputPlane", op)
        if hasattr(an, "OutputPlane"):
            monkeypatch.setattr(an, "OutputPlane", op)
        an.task9()
        op.assert_called_once()

    def test_op_setup(self, an, elements, monkeypatch):
        op = MagicMock(wraps=elements.OutputPlane)
        monkeypatch.setattr(elements, "OutputPlane", op)
        if hasattr(an, "OutputPlane"):
            monkeypatch.setattr(an, "OutputPlane", op)
        an.task9()
        op.assert_called_once()
        args = op.call_args.args
        args += tuple(op.call_args.kwargs.values())
        assert len(args) == 1
        assert args[0] == 250

    def test_ray_called(self, an, rays, monkeypatch):
        ray = MagicMock(wraps=rays.Ray)
        monkeypatch.setattr(rays, "Ray", ray)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an, "Ray", ray)
        an.task9()
        ray.assert_called()

    def test_ray_called_multiple(self, an, rays, monkeypatch):
        ray = MagicMock(wraps=rays.Ray)
        monkeypatch.setattr(rays, "Ray", ray)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an, "Ray", ray)
        an.task9()
        assert ray.call_count > 1

    def test_ray_vertices_called(self, an, rays, monkeypatch):
        vertices = MagicMock(return_value=[np.array([0.,0.,0.])])
        monkeypatch.setattr(rays.Ray, "vertices", vertices)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an.Ray, "vertices", vertices)
        an.task9()
        vertices.assert_called()

    def test_ray_vertices_called_multiple(self, an, rays, monkeypatch):
        vertices = MagicMock(return_value=[np.array([0.,0.,0.])])
        monkeypatch.setattr(rays.Ray, "vertices", vertices)
        if hasattr(an, "Ray"):
            monkeypatch.setattr(an.Ray, "vertices", vertices)
        an.task9()
        assert vertices.call_count > 1

    def test_output_fig(self, an):
        fig = an.task9()
        assert isinstance(fig, mpl.figure.Figure)

    @check_figures_equal(ref_path="task9", tol=32)
    def test_plot9(self, an):
        return an.task9()

class TestTask10:

    def test_task10_exists(self, an):
        assert hasattr(an, "task10")

    def test_focal_point(self, an):
        _, fp = an.task10()
        assert np.isclose(fp, 200.)

    def test_ouput_fig(self, an):
        fig, _ = an.task10()
        assert isinstance(fig, mpl.figure.Figure)

    @check_figures_equal(ref_path="task10", tol=32)
    def test_plot10(self, an):
        fig, _ = an.task10()
        return fig


class TestTask12:

    def test_task12_exists(self, an):
        assert hasattr(an, "task12")

    def test_ouput_fig(self, an):
        fig = an.task12()
        assert isinstance(fig, mpl.figure.Figure)

    @check_figures_equal(ref_path="task12", tol=32)
    def test_plot12(self, an):
        return an.task12()

class TestTask13:
    def test_task13_exists(self, an):
        assert hasattr(an, "task13")

    def test_rms(self, an):
        _, rms = an.task13()
        assert np.isclose(rms, 0.0020035841289414527)

    def test_output_fig(self, an):
        fig, _ = an.task13()
        assert isinstance(fig, mpl.figure.Figure)

    @check_figures_equal(ref_path="task13", tol=33)
    def test_plot13(self, an):
        fig, _ = an.task13()
        return fig

class TestTask14:

    def test_output_fig(self, an):
        fig, _ = an.task14()
        assert isinstance(fig, mpl.figure.Figure)

    def test_difraction(self, an):
        _, diffraction_scale = an.task14()
        assert np.isclose(diffraction_scale, 0.01176)

class TestTask15:

    def test_output_fig(self, an):
        fig = an.task15()
        assert isinstance(fig, mpl.figure.Figure)

    def test_2SR_objects_created(self, elements, monkeypatch):
        sr = MagicMock(wraps=elements.SphericalRefraction)
        op = MagicMock(wraps=elements.OutputPlane)
        monkeypatch.setattr(elements, "SphericalRefraction", sr)
        monkeypatch.setattr(elements, "OutputPlane", op)
        an = import_module("analysis")
        an.task15()
        assert sr.call_count >= 2
        assert op.called
