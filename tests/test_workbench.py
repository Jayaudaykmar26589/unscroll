import pytest
from scripts.build_workbench import verify_sources
from scripts.validate_mesh_reference import sha


def test_workbench_refuses_stale_geometry_and_proxy_scores(tmp_path):
    trace=tmp_path/'trace.jsonl';trace.write_text('{"points":[[1,2]]}')
    proxy=tmp_path/'proxy.json';proxy.write_text('{"ridge":0.8}')
    report={'trace_sources':[{'file':trace.name,'sha256':sha(trace)}],
            'proxy_source':{'file':proxy.name,'sha256':sha(proxy)}}
    verify_sources(tmp_path,report)
    trace.write_text('{"points":[[10,20]]}')
    with pytest.raises(ValueError,match='Trace changed'):
        verify_sources(tmp_path,report)
    report['trace_sources'][0]['sha256']=sha(trace)
    proxy.write_text('{"ridge":1}')
    with pytest.raises(ValueError,match='Proxy report changed'):
        verify_sources(tmp_path,report)
