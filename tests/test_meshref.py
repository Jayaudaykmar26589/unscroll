import numpy as np
import pytest
from src.eval.meshref import (load_obj, validate_frame, curve_xyz, sample_curve,
    paired_triangle_distance, SurfaceIndex, slice_segments)


def test_xyz_uses_physical_z_not_row_or_file_index():
    p=curve_xyz({'points':[[4820,3403]],'z':0,'source_z':6100,'source_file':'06100.tif'})
    np.testing.assert_array_equal(p,[[4820,3403,6100]])
    with pytest.raises(ValueError,match='source_z'):
        curve_xyz({'points':[[1,2]],'z':0})
    with pytest.raises(ValueError,match='disagrees'):
        curve_xyz({'points':[[1,2]],'source_z':0,'source_file':'06100.tif'})


def test_frame_requires_matching_volume_and_full_tiff_shape():
    meta={'uuid':'A','height':7888,'width':8096}
    np.testing.assert_array_equal(validate_frame({'volume':'A'},meta,(7888,8096)),np.eye(4))
    with pytest.raises(ValueError,match='IDs'):
        validate_frame({'volume':'B'},meta,(7888,8096))
    with pytest.raises(ValueError,match='crop/scale'):
        validate_frame({'volume':'A'},meta,(8096,7888))


def test_large_sparse_triangle_interior_is_not_vertex_distance():
    v=np.array([[0,0,0],[100,0,0],[0,100,0]])
    index=SurfaceIndex(v,[[0,1,2]])
    np.testing.assert_allclose(index.distances([[25,25,2],[50,50,0],[101,0,0]]),[2,0,1])


def test_exact_candidates_do_not_miss_long_triangle():
    v=np.array([[0,0,0],[1000,0,0],[0,1000,0],[0,0,3],[1,0,3],[0,1,3]])
    idx=SurfaceIndex(v,[[0,1,2],[3,4,5]])
    np.testing.assert_allclose(idx.distances([[.1,.1,.5]]),[.5])


def test_degenerate_and_outside_triangles():
    tri=np.array([[[0,0,0],[0,0,0],[2,0,0]],[[0,0,0],[2,0,0],[0,2,0]]])
    np.testing.assert_allclose(paired_triangle_distance([[1,2,0],[2,2,0]],tri),[2,2**.5])


def test_obj_negative_indices_and_quad(tmp_path):
    p=tmp_path/'q.obj';p.write_text('v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\nf -4/1 -3/2 -2/3 -1/4\n')
    v,f=load_obj(p);assert f.shape==(2,3)
    np.testing.assert_allclose(SurfaceIndex(v,f).distances([[1,1,1]]),[1])


def test_slice_interpolates_crossing_faces_without_vertices_in_window():
    v=np.array([[10,20,6000],[10,40,6200],[10,60,6000]])
    seg=slice_segments(v,[[0,1,2]],6100)
    np.testing.assert_allclose(seg,[[[10,30],[10,50]]])
    assert len(slice_segments(v,[[0,1,2]],0))==0


def test_arc_weighting_does_not_reward_dense_vertices():
    a,w=sample_curve(np.array([[0,0,0],[10,0,0]]))
    b,wb=sample_curve(np.array([[0,0,0],[.01,0,0],[.02,0,0],[10,0,0]]))
    np.testing.assert_allclose(a,b);np.testing.assert_allclose(w,wb)
    c,wc=sample_curve(np.array([[10,0,0],[0,0,0]]))
    np.testing.assert_allclose(a,c[::-1]);assert wc.sum()==10


def test_parallel_wrong_layer_stays_wrong_no_registration():
    v=np.array([[0,0,6000],[0,100,6000],[0,0,6200],[0,100,6200]])
    idx=SurfaceIndex(v,[[0,1,2],[1,3,2]])
    p=curve_xyz({'points':[[20,25],[20,75]],'source_z':6100})
    np.testing.assert_allclose(idx.distances(p),[20,20])


def test_slice_through_vertex_keeps_crossing_segment():
    v=np.array([[0,0,1],[2,0,0],[2,2,2]])
    np.testing.assert_allclose(slice_segments(v,[[0,1,2]],1),[[[0,0],[2,1]]])


def test_bvh_matches_exhaustive_triangle_distance():
    rng=np.random.default_rng(42)
    vertices=rng.normal(size=(300,3));faces=np.arange(300).reshape(-1,3)
    points=rng.normal(size=(40,3))*2
    actual=SurfaceIndex(vertices,faces).distances(points)
    expected=[paired_triangle_distance(np.broadcast_to(p,(100,3)),vertices[faces]).min() for p in points]
    np.testing.assert_allclose(actual,expected,atol=1e-10)


def test_fixed_roi_clips_reference_length_and_reports_missing():
    from scripts.validate_mesh_reference import clipped_segments,metrics,distance_to_curve
    seg=clipped_segments(np.array([[[-10,5],[20,5]],[[1,20],[2,20]]]),[[0,0],[10,10]])
    np.testing.assert_allclose(seg,[[[0,5],[10,5]]])
    assert not metrics(np.array([]),np.array([]))['assessed']
    assert metrics(np.array([0.,10.]),np.array([1.,9.]))['within_2_voxels']==.1
    np.testing.assert_allclose(distance_to_curve(np.array([[5,7]]),seg[0]),[2])
