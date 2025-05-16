import torch
import numpy as np
import os
import subprocess
from postprocess_data import postprocess
from Network import U_Net as FCNNet
from ufold_predict import get_ct_dict_fast
from ufold.data_generator import RNASSDataGenerator, Dataset,RNASSDataGenerator_input
# 假设以下是已定义的函数和类
# 这里需要根据实际情况补充完整




# 假设已经加载了预训练模型
contact_net = FCNNet(img_ch=17,output_ch=1)
MODEL_SAVED = 'models/ufold_train_pdbfinetune.pt'
contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location={"cuda:3":"cuda:0"}))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
contact_net.to(device)

# 假设输入的RNA序列
input_seq = "UGUUGUUAUGUGUUGGUUAUGUGUUGAAUAUAAUGUCCUAUAAGCUUCAGUUAGGUCAGGUUAUGUGUUGCUUAAAUCUUUUGUACCUUUACUGAUUUGUGUGAGAGAGUGUGUGUGUGUGUGUGUGUCUGUGUUUGCACGCGCACAUGUGCGUGCGUGCUUGUGCUUGUUUUAUUACUUGCUGAGAGGAUACUACAAACUCAAACAAUUAUUGUAGAUUUAGAAUUACCUUACUUAU"

# 数据准备：将序列转换为One - Hot编码
seq_embedding = one_hot_600(input_seq)
seq_embedding = torch.Tensor(seq_embedding).float().unsqueeze(0).to(device)  # 添加批次维度

seq_ori = torch.Tensor(seq_embedding).float().to(device)
seq_len = len(input_seq)

# 模型推理
with torch.no_grad():
    pred_contacts = contact_net(seq_embedding)

# 后处理
u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
map_no_train = (u_no_train > 0.5).float()

# 创建保存结果的文件夹
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/save_ct_file'):
    os.makedirs('results/save_ct_file')
if not os.path.exists('results/save_varna_fig'):
    os.makedirs('results/save_varna_fig')

# 获取CT字典和点括号表示
batch_n = 1
ct_dict_all = {}
dot_file_dict = {}
seq_name = "example_seq"
ct_dict_all, dot_file_dict, tertiary_bp = get_ct_dict_fast(map_no_train, batch_n, ct_dict_all, dot_file_dict, seq_ori.cpu().squeeze(), seq_name)

# 生成CT文件
ct_file_name = f'results/save_ct_file/{seq_name}.ct'
with open(ct_file_name, 'w') as ct_file:
    for i in range(len(ct_dict_all[1])):
        ct_file.write(f'{ct_dict_all[1][i][0]}\t{ct_dict_all[1][i][1]}\n')

# 绘制RNA结构图形
subprocess.Popen(["java", "-cp", "VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd",
                  '-i', ct_file_name,
                  '-o', f'results/save_varna_fig/{seq_name}_radiate.png',
                  '-algorithm', 'radiate',
                  '-resolution', '8.0',
                  '-bpStyle', 'lw'],
                 stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

print("处理完成，请查看results文件夹中的预测结果。")