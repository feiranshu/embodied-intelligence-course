"""Offline raw-feedback audit: select newest session per ID; never change calibration."""
from pathlib import Path
from datetime import datetime
import csv
import json
import statistics
import argparse
import itertools
import hashlib

ROOT = Path(__file__).resolve().parent


def load_session(folder):
    import io
    captured = {name:(folder/name).read_bytes() for name in ['session.json','serial_and_events.jsonl','positions.csv']}
    meta = json.loads(captured['session.json'].decode('utf-8'))
    events = [json.loads(line) for line in captured['serial_and_events.jsonl'].decode('utf-8').splitlines() if line.strip()]
    samples = list(csv.DictReader(io.StringIO(captured['positions.csv'].decode('utf-8-sig'))))
    for row in samples:
        row['sample'] = int(row['sample'])
        for k in ['request_started_s','reply_finished_s']:
            row[k] = float(row[k])
        row['raw'] = int(row['raw_position']) if row['raw_position'] else None
    return meta, events, samples, {name:hashlib.sha256(value).hexdigest() for name,value in captured.items()}


def analyze(folder, loaded=None):
    meta, events, samples, hashes = loaded or load_session(folder)
    good = [r for r in samples if r['raw'] is not None and not r['error']]
    by_sample = {r['sample']:r for r in samples}
    marks = []
    for event in events:
        if event['kind']=='operator_mark':
            row = by_sample.get(event['latest_sample'])
            marks.append(dict(elapsed_s=event['elapsed_s'], sample=event['latest_sample'],
                              raw=row['raw'] if row else None,
                              sample_age_s=event['elapsed_s']-row['reply_finished_s'] if row else None))
    jumps = []
    for a,b in zip(samples,samples[1:]):
        if a['raw'] is not None and b['raw'] is not None:
            delta = b['raw']-a['raw']
            if abs(delta)>128:
                jumps.append(dict(sample=b['sample'], elapsed_s=b['reply_finished_s'],
                                  before=a['raw'],after=b['raw'],delta=delta,
                                  dt_s=b['reply_finished_s']-a['reply_finished_s'],
                                  direction=b['command_direction']))
    # Exact constant value, same continuously commanded direction. No smoothing/deduplication.
    plateaus = []
    run = []
    def flush():
        if len(run)>1:
            duration = run[-1]['reply_finished_s']-run[0]['request_started_s']
            if duration >= .5:
                plateaus.append(dict(raw=run[0]['raw'],direction=run[0]['command_direction'],
                                     start_s=run[0]['request_started_s'],end_s=run[-1]['reply_finished_s'],
                                     duration_s=duration,samples=len(run)))
    for row in samples:
        eligible = row['raw'] is not None and not row['error'] and row['command_direction']!='stopped'
        if not eligible:
            flush(); run=[]
        elif run and (row['raw']!=run[-1]['raw'] or row['command_direction']!=run[-1]['command_direction']):
            flush(); run=[row]
        else:
            run.append(row)
    flush()
    motion = [e for e in events if e['kind'] in ['motion','stop']]
    endpoint_runs = []
    for raw, group in itertools.groupby(samples,key=lambda row:row['raw']):
        group = list(group)
        if raw in (0,1022):
            endpoint_runs.append(dict(raw=raw,start_s=group[0]['request_started_s'],
                end_s=group[-1]['reply_finished_s'],samples=len(group),
                duration_s=group[-1]['reply_finished_s']-group[0]['request_started_s'],
                command_directions=sorted(set(r['command_direction'] for r in group))))
    verified = 0
    for row in samples:
        packet = bytes.fromhex(row['reply_hex'])
        if (len(packet)==8 and packet[:2]==b'\xff\xff' and packet[2]==meta['id']
                and packet[3]==4 and packet[4]==0 and sum(packet[2:]) & 255 == 255
                and ((packet[5]<<8)|packet[6])==row['raw']):
            verified += 1
    span_samples = []
    if len(marks)>=2:
        span_samples = [r for r in good if marks[0]['sample']<=r['sample']<=marks[-1]['sample']]
    dts = [b['reply_finished_s']-a['reply_finished_s'] for a,b in zip(samples,samples[1:])]
    return dict(id=meta['id'],source=folder.name,started_utc=meta['started_utc'],
                session_complete=any(e['kind']=='session_end' for e in events),
                source_sha256=hashes,
                samples=len(samples),errors=sum(bool(r['error']) for r in samples),
                raw_min=min(r['raw'] for r in good),raw_max=max(r['raw'] for r in good),
                marks=marks,motion_events=motion,jumps_over_128_counts=jumps,
                packet_and_position_verified_rows=verified,endpoint_constant_runs=endpoint_runs,
                constant_commanded_runs_at_least_0_5_s=plateaus,
                median_sample_interval_s=statistics.median(dts),max_sample_interval_s=max(dts),
                mean_poll_hz=(len(samples)-1)/(samples[-1]['reply_finished_s']-samples[0]['reply_finished_s']),
                marked_interval_raw_min=min(r['raw'] for r in span_samples) if span_samples else None,
                marked_interval_raw_max=max(r['raw'] for r in span_samples) if span_samples else None,
                stop_confirmed=meta.get('stop_confirmed'),mode_restored=meta.get('original_mode_restored'))


def write_report(result, output):
    lines = ['# 最新关节反馈记录分析（2026-10-16）', '',
        f"分析快照时间：{result['analyzed_utc']}。以各ID的session.started_utc选择最新记录。", '',
        '## 1. 核心结论', '',
        '**操作者本次明确观察到：ID5真实关节继续转动时，返回值仍保持1022。**', '',
        '这项现场观察与日志中的端点平台相互支持：ID5在该区域内的反馈不能唯一确定真实角度。同一个1022对应多个实际轴角，简单取模展开或调整限位无法恢复丢失的角度信息。异常区的精确角宽仍不能从现有日志确定，不能直接认定为恰好60°。', '',
        '## 2. 所选会话与M标记', '',
        '| ID | 最新会话 | 采样数 | 全部M标记读数（按时间） | 会话结束记录 |',
        '|---|---|---:|---|---|']
    for j in result['joints']:
        marks = ' → '.join(str(m['raw']) for m in j['marks']) or '无标记'
        lines.append(f"| {j['id']} | {j['source']} | {j['samples']} | {marks} | {'有' if j['session_complete'] else '未发现'} |")
    lines += ['', 'M标记使用按键时最新一次采样。它是操作者指定的物理端点/观察点，不自动等价于相隔360°。若同一会话有多个标记，全部保留，不擅自猜测哪两个是不同端点。', '',
        '排除旧会话：'+ '、'.join(result['excluded_sessions'])+'。', '', '## 3. 数据核验', '',
        f"当前选取记录共{sum(j['samples'] for j in result['joints'])}行；位置读取错误{sum(j['errors'] for j in result['joints'])}次。",
        f"独立检查帧头、ID、长度、状态码、校验和及高字节在前的位置解码，共{sum(j['packet_and_position_verified_rows'] for j in result['joints'])}行与CSV一致。",
        '分析结果与轨迹图使用同一份读入快照，源文件SHA256保存在analysis.json。原始记录未修改。', '',
        '## 4. 原始反馈的跨端点行为', '',
        '| ID | 观测范围 | 大跳变前后（相邻读数差超过128，仅用于描述） | 0/1022连续不变段 |',
        '|---|---|---|---|']
    for j in result['joints']:
        jumps = '；'.join(f"{x['before']}→{x['after']}" for x in j['jumps_over_128_counts']) or '未观察到'
        runs = '；'.join(f"{x['raw']}保持{x['duration_s']:.3f}秒" for x in j['endpoint_constant_runs'] if x['duration_s']>=.1) or '未观察到'
        lines.append(f"| {j['id']} | {j['raw_min']}～{j['raw_max']} | {jumps} | {runs} |")
    lines += ['', '上述平台时长取连续相同读数的首个请求至末个回复，不是机械角度。端点平台、反向小值及大跳变都保留在原始CSV，分析没有做平滑、去重或角度补偿。', '',
        '例如ID1在left指令持续期间，6→463→1022两次相邻变化约间隔31、33毫秒，状态包都校验通过。旧PositionTrace按1024取模，得到约+457和-465，会触发其超过128格的检查。取消检查也不能把这些值自动变成可靠角度。', '',
        'ID2整份日志未出现motion事件；ID3约14.19秒后无进一步转动命令而反馈仍变化。程序记录的stopped只表示未由本程序下发转动命令，不能确定手动、重力或其他运动原因，也不能据此认定舵机自行运动。', '',
        '## 5. 标记完整性与停止状态', '']
    for j in result['joints']:
        if not j['session_complete']:
            lines.append(f"- ID{j['id']}：此快照未发现session_end，无法确认最终停止和模式恢复；不能把未结束的会话摘要视为正常退出。")
        if len(j['marks'])<2:
            lines.append(f"- ID{j['id']}：少于两个M标记，不能确认两侧端点。")
        elif len(set(m['raw'] for m in j['marks']))==1:
            lines.append(f"- ID{j['id']}：目前全部M标记均为{j['marks'][0]['raw']}。不能擅自把这组相同值解释为已确定的两个不同端点；可能重复标记同一点，也可能体现反馈歧义，需要结合实际标记位置。")
        if j['session_complete']:
            lines.append(f"- ID{j['id']}：最终停止确认={j['stop_confirmed']}，模式恢复确认={j['mode_restored']}。")
    lines += ['', '## 6. 对FK及手眼标定的判断', '',
        '串口正确收到1022，不代表1022能唯一代表实际机械角度。操作者已确认ID5存在转动中保持1022的区域，因此现有全范围FK的输入角度可靠性不成立；该区域的完整姿态不能只靠当前反馈恢复。', '',
        '这为旧手眼采集中ID5反复出现1022、ID1反复出现0提供了具体风险依据，但新记录没有与旧照片同步，不能据此给旧12张照片逐张恢复真实角度，或排除照片同步与相机固定问题。', '',
        'M给出的机械活动端点与传感器可靠测角区间不同；不应把端点之间所有读数自动线性映射为360°。速度参数也不是独立角速度基准，不能用平台持续时间直接推角度。', '',
        '## 7. 下一步', '',
        'ID5平台现象无需再靠重复采样来证明。若要得到该段真实角度，需要额外的角度观测，例如同步视频中带明确几何参考的角度、外部角度测量或其他传感器；也可对经过验证的单调测量区间单独校准参考角与方向，再评价该区间的FK。', '',
        '本轮未修改advanced.json、FK零角/方向、运动限位或旧手眼矩阵，未新增控制限制。', '',
        '## 文件', '',
        '- [最新原始轨迹图](feedback_traces.png)：橙色阴影为0/1022保持至少0.1秒的采样段，只用于呈现。',
        '- [完整分析结果](analysis.json)：选取依据、哈希、标记、跳变、端点平台和校验计数。',
        '- 分析程序：`../../analyze_rotation_records.py`。在其所在目录运行 `python .\\analyze_rotation_records.py` 可复算，需要matplotlib，不访问硬件。']
    (output/'反馈分析报告_20261016.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,default=ROOT/'rotation_records')
    parser.add_argument('--output',type=Path,default=ROOT/'rotation_records'/'analysis_20261016')
    args = parser.parse_args()
    latest = {}; excluded = []
    for folder in args.records.iterdir():
        if not folder.is_dir() or not (folder/'session.json').is_file():
            continue
        meta = json.loads((folder/'session.json').read_text(encoding='utf-8'))
        stamp = datetime.fromisoformat(meta['started_utc'])
        old = latest.get(meta['id'])
        if old is None or stamp>old[0]:
            if old: excluded.append(old[1].name)
            latest[meta['id']] = (stamp,folder)
        else:
            excluded.append(folder.name)
    cached = {i:load_session(latest[i][1]) for i in latest}
    result = dict(analyzed_utc=datetime.now().astimezone().isoformat(),
                  selection='newest session started_utc per servo ID',excluded_sessions=sorted(excluded),
                  missing_ids=[i for i in range(1,7) if i not in latest],
                  operator_observation='ID5 physically continues rotating while returned position stays at 1022 (user confirmed)',
                  joints=[analyze(latest[i][1],cached[i]) for i in sorted(latest)])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(result,args.output)
    # Static scientific time-series plot, actual raw values with no unwrapping.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','DejaVu Sans']
    fig, axes = plt.subplots(len(latest),1,figsize=(13,3.2*len(latest)),layout='constrained')
    for ax, joint in zip(axes,result['joints']):
        _,events,rows,_ = cached[joint['id']]
        ax.plot([r['reply_finished_s'] for r in rows],[r['raw'] for r in rows],lw=1.2,color='#185da8',label='原始反馈')
        for index,mark in enumerate(joint['marks'],1):
            ax.axvline(mark['elapsed_s'],color='#b33c00',ls='--',lw=1)
            ax.scatter([mark['elapsed_s']],[mark['raw']],color='#b33c00',zorder=5)
            ax.annotate(f"M{index}={mark['raw']}",(mark['elapsed_s'],mark['raw']),
                        xytext=(5,10 if mark['raw']<800 else -20),textcoords='offset points',fontsize=10)
        for run in joint['endpoint_constant_runs']:
            if run['duration_s']>=.1:
                ax.axvspan(run['start_s'],run['end_s'],color='#e8a440',alpha=.25)
        ax.set(title=f"ID{joint['id']} · {joint['source']}",xlabel='会话时间 / 秒',ylabel='原始读数')
        ax.set_ylim(-35,1060); ax.set_yticks([0,256,512,768,1023]); ax.grid(alpha=.2)
    fig.suptitle('各关节最新记录：M为人工端点标记；橙色阴影为0或1022读数连续不变≥0.1秒',fontsize=14)
    fig.savefig(args.output/'feedback_traces.png',dpi=150)
    plt.close(fig)
    print(json.dumps(dict(analyzed_at=result['analyzed_utc'],total_samples=sum(j['samples'] for j in result['joints']),
        selected=[dict(id=j['id'],source=j['source'],samples=j['samples'],complete=j['session_complete'],marks=[m['raw'] for m in j['marks']]) for j in result['joints']]),ensure_ascii=False))


if __name__=='__main__':
    main()
