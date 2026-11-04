import numpy as np

from src.trace.image_ridge import trace_image_ridge, estimate_motion
from src.trace.guards import LayerGuard, _project, _polyline, check_curvature


def scene(dx=0, dy=0, mask_center=78):
    y, x = np.indices((170,180))
    ridge = 70 + 4*np.sin((y-dy)/25) + dx
    image = 100 + 1000*np.exp(-((x-ridge)/1.5)**2)
    image += 800*np.exp(-((x-ridge-20)/1.5)**2)
    image += np.random.default_rng(1).normal(0,5,image.shape)
    mask = (abs(x-mask_center) < 23) & (y>=10) & (y<=155)
    guide_y=np.arange(20.,145.)
    guide=np.column_stack((np.full(len(guide_y),mask_center),guide_y))
    return image, mask, guide


def test_seeded_path_follows_ct_ridge_not_broad_mask_medial_axis():
    image,mask,guide=scene()
    p,info=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    assert info["success"]
    true_x=70+4*np.sin(p[:,1]/25)
    assert np.max(abs(p[:,0]-true_x)) < 1.5
    assert np.linalg.norm(np.diff(p,axis=0),axis=1).max() < 3
    assert check_curvature(p)["pass"]


def test_seed_pinning_does_not_introduce_a_single_sample_kink():
    image,mask,guide=scene()
    image[79:82,69:73] += 3000
    p,info=trace_image_ridge(image,mask,[1,1,0,0],guide,[72,80])
    assert info["success"]
    anchor=np.asarray(info["seed_anchor"])
    row=np.argmin(abs(p[:,1]-anchor[1]))
    np.testing.assert_allclose(p[row],anchor)
    assert check_curvature(p)["pass"]


def test_mask_center_jump_cannot_change_layer_in_static_ct():
    image,mask,guide=scene()
    p,_=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    _,other,other_guide=scene(mask_center=62)
    q,info=trace_image_ridge(image,other,[1,1,0,0],other_guide,[71,80],p,image)
    assert info["success"] and info["motion"]["verified"]
    assert (info["motion"]["dx"],info["motion"]["dy"]) == (0,0)
    distance,_=_project(q,_polyline(p)[0])
    assert distance.max() <= 1.5


def test_image_motion_is_measured_independently_and_guard_uses_matching_reference():
    image,mask,guide=scene()
    p,_=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    next_image,next_mask,next_guide=scene(dx=1,dy=-1)
    q,info=trace_image_ridge(next_image,next_mask,[1,1,0,0],next_guide,[71,80],p,image)
    assert info["success"]
    assert (info["motion"]["dx"],info["motion"]["dy"]) == (1,-1)
    guard=LayerGuard()
    guard.evaluate(p,0)
    result=guard.evaluate(q,1,image_motion={**info["motion"],"reference_z":0})
    assert result["status"] == "ok" and result["image_motion_used"]


def test_static_image_cannot_register_away_a_parallel_curve_jump():
    image,mask,guide=scene()
    p,_=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    motion=estimate_motion(image,image,p)
    guard=LayerGuard()
    guard.evaluate(p,0)
    result=guard.evaluate(p+[20,0],1,image_motion={**motion,"reference_z":0})
    assert result["winding"]["pass"] is False


def test_weak_or_wrong_reference_motion_is_not_applied_by_guard():
    image,mask,guide=scene()
    p,_=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    for score,reference in ((.2,0),(.95,99)):
        guard=LayerGuard();guard.evaluate(p,0)
        result=guard.evaluate(p+[20,0],1,image_motion={"method":"ct_ncc","verified":True,
            "score":score,"reference_z":reference,"dx":20,"dy":0,"search_radius":30})
        assert not result["image_motion_used"]
        assert result["winding"]["pass"] is False


def test_missing_ct_or_sam_support_fails_instead_of_copying_reference():
    image,mask,guide=scene()
    p,_=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    blank=np.zeros_like(image)
    q,info=trace_image_ridge(blank,mask,[1,1,0,0],guide,[71,80],p,image)
    assert len(q)==0 and not info["success"]
    q,info=trace_image_ridge(image,np.zeros_like(mask),[1,1,0,0],guide,[71,80],p,image)
    assert len(q)==0 and not info["success"]


def test_partial_sam_endpoint_loss_keeps_ct_supported_extent_without_reanchoring():
    image,mask,guide=scene()
    p,_=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    shortened=mask.copy()
    shortened[:40]=False
    q,info=trace_image_ridge(image,shortened,[1,1,0,0],guide,[71,80],p,image)
    assert info["success"]
    assert .5 < info["sam_overlap_fraction"] < 1
    assert len(q)==len(p)
    assert np.max(np.linalg.norm(q-p,axis=1)) < 1.5


def test_folded_patch_is_not_silently_collapsed_to_monotone_path():
    image,mask,_=scene()
    guide=np.column_stack((np.r_[np.full(70,60),np.full(70,80)],
                            np.r_[np.arange(30,100),np.arange(100,30,-1)]))
    q,info=trace_image_ridge(image,mask,[1,1,0,0],guide,[71,80])
    assert len(q)==0 and info["reason"] == "non_monotone_patch"


def test_faint_moving_ridge_does_not_lag_behind_static_bright_neighbor():
    # The current image, not a soft attraction to the old curve, must choose
    # position inside the hard corridor. The previous proximity term caused
    # >2 voxels of accumulated lag in this ten-frame known-geometry case.
    y,x=np.indices((170,180))
    center=70+15*np.sin(y/25)
    mask=(abs(x-75)<35)&(y>=10)&(y<=155)
    gy=np.arange(20.,145.)
    guide=np.column_stack((70+15*np.sin(gy/25),gy))
    def image(shift):
        return 100+180*np.exp(-((x-center-shift)/1.5)**2)+1000*np.exp(-((x-center-20)/1.5)**2)
    previous=guide.copy()
    previous_image=image(0)
    for frame in range(1,11):
        current=image(frame*.4)
        path,info=trace_image_ridge(current,mask,[1,1,0,0],guide,[71,80],previous,previous_image)
        assert info["success"]
        truth=70+15*np.sin(path[:,1]/25)+frame*.4
        assert np.mean(abs(path[:,0]-truth)) < .6
        previous,previous_image=path,current
