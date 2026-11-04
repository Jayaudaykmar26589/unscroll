"""Released surfaces in volume voxel XYZ, never fitted to predicted curves.

OBJ positions are (column, row, physical slice). TIFF arrays are [row, column].
Local trace file indices are NOT volume z. Volume IDs must match before scoring.
"""

import numpy as np


def load_obj(path):
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) for p in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for j in range(1, len(idx)-1):
                    faces.append([idx[0], idx[j], idx[j+1]])
    v = np.asarray(verts, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    if not np.isfinite(v).all() or (len(f) and (f.min() < 0 or f.max() >= len(v))):
        raise ValueError('Invalid OBJ geometry')
    return v, f


def validate_frame(segment_meta, volume_meta, image_shape):
    """Identity transform is valid only for the explicitly linked volume."""
    if not volume_meta.get('uuid') or segment_meta.get('volume') != volume_meta['uuid']:
        raise ValueError('Mesh and CT volume IDs do not match')
    if tuple(image_shape) != (volume_meta['height'], volume_meta['width']):
        raise ValueError('CT dimensions differ: an explicit crop/scale transform is required')
    return np.eye(4)


def curve_xyz(record):
    """Serialized full-resolution XY plus source_z; fail closed on missing z."""
    if 'source_z' not in record:
        raise ValueError('source_z is required; local z is not a volume coordinate')
    p = np.asarray(record['points'], dtype=float).reshape(-1, 2)
    z = float(record['source_z'])
    if not np.isfinite(p).all() or not np.isfinite(z):
        raise ValueError('Nonfinite curve')
    name = record.get('source_file', '').split('.')[0]
    if name.isdigit() and int(name) != z:
        raise ValueError('source_z disagrees with TIFF filename')
    return np.column_stack([p, np.full(len(p), z)])


def sample_curve(points, spacing=1.):
    """Arc-length midpoint quadrature; invariant to vertex density/direction."""
    p = np.asarray(points, float)
    if len(p) < 2: return np.empty((0, p.shape[-1])), np.empty(0)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.r_[True, lengths > 1e-12]; p = p[keep]
    if len(p) < 2: return np.empty((0, p.shape[-1])), np.empty(0)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(1, int(np.ceil(s[-1]/spacing)))
    q = (np.arange(n)+.5)*s[-1]/n
    return np.column_stack([np.interp(q,s,p[:,i]) for i in range(p.shape[1])]), np.full(n,s[-1]/n)


def paired_triangle_distance(points, triangles):
    """Exact Euclidean point-to-triangle distance, including degenerate faces."""
    p=np.asarray(points,float);t=np.asarray(triangles,float)
    a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a;ap=p-a
    dot=lambda x,y:np.einsum('ij,ij->i',x,y)
    d00=dot(ab,ab);d01=dot(ab,ac);d11=dot(ac,ac)
    d20=dot(ap,ab);d21=dot(ap,ac);den=d00*d11-d01*d01
    safe=np.where(den>1e-20,den,1)
    u=(d11*d20-d01*d21)/safe;v=(d00*d21-d01*d20)/safe
    inside=(den>1e-20)&(u>=0)&(v>=0)&(u+v<=1)
    proj=a+u[:,None]*ab+v[:,None]*ac
    best=np.where(inside,dot(p-proj,p-proj),np.inf)
    for x,y in ((a,b),(b,c),(c,a)):
        edge=y-x;length=dot(edge,edge)
        f=np.clip(dot(p-x,edge)/np.maximum(length,1e-30),0,1)
        delta=p-(x+f[:,None]*edge)
        best=np.minimum(best,dot(delta,delta))
    return np.sqrt(np.maximum(best,0))


class SurfaceIndex:
    """Exact triangle BVH with bounded batches; NumPy only, CPU only."""
    def __init__(self, vertices, faces):
        self.tri=np.asarray(vertices)[np.asarray(faces)]
        if not len(self.tri): raise ValueError('Surface has no triangles')
        self.centers=self.tri.mean(1)
        def build(ids):
            tri=self.tri[ids];lo=tri.min((0,1));hi=tri.max((0,1))
            if len(ids)<=24:return (lo,hi,ids,None)
            axis=int(np.ptp(self.centers[ids],axis=0).argmax())
            order=ids[np.argsort(self.centers[ids,axis])];mid=len(order)//2
            return (lo,hi,build(order[:mid]),build(order[mid:]))
        self.root=build(np.arange(len(self.tri)))

    def distances(self, points):
        p=np.asarray(points,float).reshape(-1,3)
        if not len(p):return np.empty(0)
        out=np.full(len(p),np.inf)
        for start in range(0,len(p),128):
            q=p[start:start+128];best=np.full(len(q),np.inf)
            stack=[(self.root,np.arange(len(q)))]
            while stack:
                node,ix=stack.pop();lo,hi,left,right=node
                d=np.maximum(np.maximum(lo-q[ix],q[ix]-hi),0)
                ix=ix[np.einsum('ij,ij->i',d,d)<=best[ix]**2+1e-12]
                if not len(ix):continue
                if right is None:
                    t=self.tri[left];n=len(t)
                    dist=paired_triangle_distance(np.repeat(q[ix],n,axis=0),np.tile(t,(len(ix),1,1)))
                    best[ix]=np.minimum(best[ix],dist.reshape(len(ix),n).min(1))
                else:
                    center=q[ix].mean(0)
                    children=sorted((left,right),key=lambda c:np.linalg.norm(np.maximum(np.maximum(c[0]-center,center-c[1]),0)),reverse=True)
                    stack.extend((c,ix) for c in children)
            out[start:start+len(q)]=best
        return out


def slice_segments(vertices, faces, z):
    """Intersect triangle surfaces with physical z (not nearest mesh vertices).

    Coplanar faces contribute their boundary edges; duplicate edges are removed.
    """
    tri=np.asarray(vertices)[np.asarray(faces)]
    tri=tri[(tri[:,:,2].min(1)<=z)&(tri[:,:,2].max(1)>=z)]
    segments=[]
    for t in tri:
        hits=[]
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            if a[2]==z and b[2]==z:
                segments.append([a[:2],b[:2]])
            elif min(a[2],b[2])<=z<=max(a[2],b[2]) and a[2]!=b[2]:
                hits.append((a+(b-a)*(z-a[2])/(b[2]-a[2]))[:2])
        if hits:
            hits=np.unique(np.round(hits,10),axis=0)
        if len(hits)>=2 and np.linalg.norm(hits[0]-hits[-1])>1e-9:
            segments.append([hits[0],hits[-1]])
    unique={}
    for a,b in segments:
        if np.linalg.norm(a-b)>1e-9:
            key=tuple(sorted((tuple(np.round(a,8)),tuple(np.round(b,8)))))
            unique[key]=[a,b]
    return np.asarray(list(unique.values()),float).reshape(-1,2,2)


def dense_sample(verts, faces, per_face=8, seed=0):
    """Barycentric oversample of triangles -> dense (M,3) point cloud."""
    rng = np.random.default_rng(seed)
    tris = verts[faces]
    out = np.empty((len(tris) * per_face, 3))
    for i, t in enumerate(tris):
        r1 = np.sqrt(rng.random(per_face))[:, None]
        r2 = rng.random(per_face)[:, None]
        out[i * per_face:(i + 1) * per_face] = (
            (1 - r1) * t[0] + r1 * (1 - r2) * t[1] + r1 * r2 * t[2])
    return out


def window_points(obj_path, z0, z1, per_face=8):
    """Dense reference points whose z falls in [z0, z1]."""
    verts, faces = load_obj(obj_path)
    pts = dense_sample(verts, faces, per_face)
    return pts[(pts[:, 2] >= z0) & (pts[:, 2] <= z1)]
