import torch

chkpt_final = torch.load("checkpoints/bsp_3.1m_v3/belief_space_policy_final.pth", map_location='cpu',weights_only=False)
chkpt_100 = torch.load("checkpoints/bsp_3.1m_v3/belief_space_policy_epoch_100.pth", map_location='cpu',weights_only=False)

final_sd = chkpt_final['model_state_dict']
ep100_sd = chkpt_100['model_state_dict']

are_equal = True
if final_sd.keys() != ep100_sd.keys():
    print("State dict keys differ!")
    are_equal = False
else:
    for key in final_sd:
        if not torch.equal(final_sd[key], ep100_sd[key]):
            print(f"Weights differ for key: {key}")
            are_equal = False
            # break # Optionally stop after first difference
if are_equal:
    print("Model state dicts are identical.")
else:
    print("Model state dicts differ.")