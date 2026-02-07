#!/usr/bin/env python3
"""Analyze HLO files to find remaining non-NKI dot operations."""
import json, base64, glob
from torch_neuronx.pyhlo.hlo_pb2 import HloModuleProto

def analyze_hlo(path, name):
    with open(path, "rb") as f:
        data = f.read()
    proto = HloModuleProto()
    proto.ParseFromString(data)

    dot_ops = []
    nki_ops = []
    nki_total_mac = 0
    dot_total_mac = 0

    for comp in proto.computations:
        imap = {i.id: i for i in comp.instructions}
        for inst in comp.instructions:
            if inst.opcode == "dot":
                lhs = imap[inst.operand_ids[0]].shape
                rhs = imap[inst.operand_ids[1]].shape
                dnums = inst.dot_dimension_numbers

                lc = 1
                lb = 1
                lnc = 1
                rnc = 1
                for i in range(len(lhs.dimensions)):
                    if i in dnums.lhs_contracting_dimensions: lc *= lhs.dimensions[i]
                    elif i in dnums.lhs_batch_dimensions: lb *= lhs.dimensions[i]
                    else: lnc *= lhs.dimensions[i]
                for i in range(len(rhs.dimensions)):
                    if i not in dnums.rhs_contracting_dimensions and i not in dnums.rhs_batch_dimensions:
                        rnc *= rhs.dimensions[i]
                mac = lb * lnc * lc * rnc
                dot_total_mac += mac
                ldims = [lhs.dimensions[i] for i in range(len(lhs.dimensions))]
                rdims = [rhs.dimensions[i] for i in range(len(rhs.dimensions))]
                dot_ops.append((inst.name, ldims, rdims, mac))
            elif inst.opcode == "custom-call" and inst.custom_call_target == "AwsNeuronCustomNativeKernel":
                try:
                    cfg = json.loads(base64.b64decode(inst.backend_config))
                    mac = int(cfg.get("mac_count", 0))
                except:
                    mac = 0
                nki_total_mac += mac
                nki_ops.append((inst.name, mac))

    total = nki_total_mac + dot_total_mac
    ratio = nki_total_mac / total if total > 0 else 0
    print(f"\n=== {name} ===")
    print(f"NKI ops: {len(nki_ops)}, total MACs: {nki_total_mac:,}")
    print(f"Dot ops: {len(dot_ops)}, total MACs: {dot_total_mac:,}")
    print(f"NKI ratio: {ratio:.6f}")
    print(f"\nRemaining dot ops (non-NKI):")
    for op_name, lhs_dims, rhs_dims, mac in sorted(dot_ops, key=lambda x: -x[3]):
        print(f"  {op_name}: {lhs_dims} x {rhs_dims} = {mac:,} MACs")

ce_paths = glob.glob("/tmp/nxd_model/context_encoding_model/_tp0_bk0/model.MODULE_*.hlo_module.pb")
tg_paths = glob.glob("/tmp/nxd_model/token_generation_model/_tp0_bk0/model.MODULE_*.hlo_module.pb")

if ce_paths:
    analyze_hlo(ce_paths[0], "Context Encoding")
if tg_paths:
    analyze_hlo(tg_paths[0], "Token Generation")
