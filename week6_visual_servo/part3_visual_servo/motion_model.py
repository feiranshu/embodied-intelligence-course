"""Use the hand-eye result's matching FK/IK in memory, preserving physical bounds."""


def matching_motion_model(model, result):
    if result.get('calibration_unwrap_references'):
        raise ValueError('此手眼结果使用离线角度展开，尚未验证与实机 FK/IK 一致')
    mapping = result.get('calibrated_joint_mapping')
    if mapping is None:
        return model
    return model.with_joint_calibration(mapping)
