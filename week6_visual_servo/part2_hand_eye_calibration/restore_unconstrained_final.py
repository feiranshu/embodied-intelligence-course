"""Withdraw the unreliable distance assumption and publish the best tested free fit."""
from pathlib import Path
import hashlib
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import read_json,write_json


def main():
    source=HERE/'final_calibration_20261016_previous'
    target=HERE/'最终标定结果'
    archive=HERE/'最终标定结果_已撤销12cm约束'
    result=read_json(source/'hand_eye.json')
    if result.get('distance_measurement'):
        raise ValueError('Source must be the unconstrained fit')
    result['calibration_dataset'].update(
        label='上一轮ID1基本固定的12组蓝圈样本（无距离约束最终经验解）',
        manifest_sha256=hashlib.sha256((source/'inputs/samples.json').read_bytes()).hexdigest(),
        selection='lowest all-corner training RMS among previously tested unconstrained candidates; all samples retained')
    result['assessment'].update(status='final_empirical_fit_unconstrained',
        independent_accuracy_validated=False,camera_distance_independently_measured=False,
        note='User withdrew the approximate 120 mm measurement because its reference origin was unknown. No distance constraint is used. Translation remains unvalidated; embedded empirical FK mapping required.')
    result['withdrawn_assumptions']=[dict(distance_mm=120,reason='User reported measurement unreliable and reference origin unknown',used_in_this_solution=False)]
    if target.exists():
        if archive.exists():raise FileExistsError('Distance archive already exists; inspect before repeating')
        for path in [target,archive]:
            path.resolve().relative_to(HERE.resolve())
            if path.resolve()==HERE.resolve():raise ValueError('Invalid archive path')
        target.rename(archive)
        old=archive/'README.md'
        old.write_text('> 本120 mm约束版本已撤销：用户确认距离测量不可靠。当前最终版本见旁边“最终标定结果”文件夹。\n\n'+old.read_text(encoding='utf-8'),encoding='utf-8')
        for name in ['hand_eye.json','hand_eye.png']:
            if (HERE/name).exists():shutil.copy2(HERE/name,archive/('previous_root_'+name))
    shutil.copytree(source,target)
    shutil.copy2(HERE/'README_unconstrained_final.md',target/'README.md')
    for folder in [target,HERE]:
        write_json(folder/'hand_eye.json',result)
        shutil.copy2(source/'hand_eye.png',folder/'hand_eye.png')
    print('Published unconstrained final:',result['rms_px_selected'])


if __name__=='__main__':main()
