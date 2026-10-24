"""Use the hand-eye result's matching FK/IK in memory, preserving physical bounds."""


def matching_motion_model(model, result):
    mapping = result.get('calibrated_joint_mapping')
    if mapping is None:
        return model
    return model.with_joint_calibration(mapping)
