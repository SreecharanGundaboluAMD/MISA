{
  "instruction": "DS_LOAD_TR16_B128 / GLOBAL_LOAD_TR16_B128",
  "source_file": "/home/sgundabo/MISA/amd-instinct-cdna5-instruction-set-architecture.md",
  "findings": {
    "1_instruction_format_and_operands": {
      "summary": "DS_LOAD_TR16_B128 is the LDS variant; GLOBAL_LOAD_TR16_B128 (opcode 87) is the global-memory variant. The DS section (§11.2.4, line 6627-6634) states: 'These instructions allow matrix data to be copied from LDS to VGPRs and transpose the data on the way. These are similar to the global transpose ops.' The global section (§10.9.2, line 5950) states: 'All fields of these instructions are identical to GLOBAL_LOAD_B64 and _B128, and as loads they are tracked with LOADcnt.'",
      "encoding_fields": "Identical to GLOBAL_LOAD_B128 (global) or DS_LOAD_B128 (LDS). Global variant uses VGLOBAL encoding: VADDR (address/offset VGPR), VDST (destination VGPR), SADDR (optional SGPR base), IOFFSET (24-bit signed byte offset), SO (scale-offset enable). LDS variant uses DS encoding: ADDR (address VGPR), VDST (destination VGPR), OFFSET0/OFFSET1 (8-bit immediate byte offsets).",
      "result": "Loads into 4 consecutive VGPRs, writing 128 bits per lane × 32 lanes = 4096 bits = 512 bytes = 256 × 16-bit elements (a 16×16 matrix).",
      "wave_requirement": "Wave32-only. If EXEC==0, acts like S_NOP; otherwise EXEC must be all-ones or behavior is undefined. For the DS variant, EXEC mask is ignored entirely (treated as all-ones)."
    },
    "2_transpose_addressing": {
      "concept": "The instruction loads a 16×16 matrix tile of 16-bit data where the memory layout has the OPPOSITE major order (row vs. column) from what the WMMA VGPR layout requires, and transposes on the fly.",
      "memory_layout_formulas": "Row Major (A-matrix MxK): Memory_address = (col# + row# * K) * ElementSize. Column Major (B-matrix KxN): Memory_address = (col# * K + row#) * ElementSize. (lines 5920-5923)",
      "per_lane_load": "Each of the 32 lanes loads a 128-bit contiguous chunk (8 × 16-bit elements) from memory. The doc states (line 5929-5930): 'The diagrams below show which matrix element each of the 32 lanes in a wave32 loads in a A-matrix. E.g. lane 0 loads 64 bits of contiguous memory and stores it in the matrix: K=0, M=0..7.' (The 64-bit example is for 8-bit data; for TR16_B128 it is 128 bits = 8 × 16-bit elements, same K=0, M=0..7.)",
      "per_lane_address_formula_derivation": "For an A-matrix loaded from COLUMN-MAJOR memory into ROW-MAJOR VGPRs: element[M][K] is at base + (K*16 + M)*2. Each lane reads 8 contiguous M-values for a fixed K. The per-lane source address is: lane_addr = base + (lane_id % 16) * 32 + (lane_id / 16) * 16. Where: (lane_id % 16) selects K (0..15), multiplied by 32 bytes (16 elements × 2 bytes = column stride); (lane_id / 16) selects the M-half (0 → M=0..7, 1 → M=8..15), multiplied by 16 bytes (8 elements × 2 bytes). Each lane reads 16 bytes contiguously from lane_addr. The base address comes from VADDR (global: GV mode = INST_OFFSET + VADDR[63:0], or GVS mode = INST_OFFSET + SADDR[63:0] + VADDR[31:0]) or from ADDR + offset (LDS).",
      "transpose_mechanism": "The hardware reads column-major contiguous data per lane (fixed K, consecutive M) and performs a cross-lane transpose so the result lands in VGPRs in row-major layout (fixed M across lanes, consecutive K within each lane's VGPRs). The VADDR provides the tile base address; the hardware internally generates per-lane addresses — the programmer does not supply per-lane addresses for the tile elements."
    },
    "3_offset_field_requirements": {
      "global_variant_offset": "IOFFSET is a 24-bit signed byte offset added to the base address (same as GLOBAL_LOAD_B128).",
      "lds_variant_offset": "OFFSET0 and OFFSET1 are two 8-bit fields. For single-address instructions they combine into a 16-bit unsigned byte offset {offset1, offset0} (line 6421-6423). For 2-address instructions they are used as two separate 8-bit unsigned offsets, each multiplied by 4 for 8/16/32-bit data or by 8 for 64-bit data.",
      "alignment": "The doc does not state any special alignment or stride requirement specific to the transpose instructions beyond the standard global/LDS load alignment. The base address for the tile must be naturally sized for a B128 load (the tile is 512 bytes = 16×16×2). Standard memory alignment rules apply (the doc notes for scalar loads: 'loads of 16-bit data force the address to 2-byte alignment'; B128 loads are typically DWORD-or-larger aligned). No explicit 'stride' field exists for these instructions — the stride is implicitly the matrix dimension (16 elements) baked into the hardware address generation."
    },
    "4_lane_to_element_mapping": {
      "overview": "32 lanes map onto the 16×16 matrix (256 elements). Each lane holds 128 bits = 8 sixteen-bit elements. The mapping differs for A-matrix vs B-matrix loads.",
      "a_matrix_mapping": "A-matrix (MxK, 16×16), loading from column-major memory into row-major VGPRs: Lanes 0-15 → K = lane_id (0..15), M = 0..7 (first 8 rows). Lanes 16-31 → K = lane_id - 16 (0..15), M = 8..15 (second 8 rows). Each lane reads 8 contiguous elements (fixed K, 8 consecutive M values) from column-major memory.",
      "b_matrix_mapping": "B-matrix (KxN, 16×16), the doc states (line 5934): 'one lane loads multiple contiguous N-values along single K-dimension index.' So each lane holds a fixed K and contiguous N-values — the transpose of the A-matrix pattern.",
      "result_vgpr_layout": "After transpose, the result is in row-major VGPR layout (16-bit A-matrix 16×16, 4 VGPRs): VGPR 0, Lane 0 (M=0): K=0 in bits[15:0], K=1 in bits[31:16]. VGPR 1, Lane 0 (M=0): K=2, K=3. VGPR 2, Lane 0 (M=0): K=4, K=5. VGPR 3, Lane 0 (M=0): K=6, K=7. VGPR 0, Lane 1 (M=1): K=0, K=1. ... VGPR 0, Lane 15 (M=15): K=0, K=1. Lanes 16-31 hold K=8..15 for the same M values. (Derived from the 16-bit A-Matrix 16×32 layout table at lines 4471-4479, halved for 16×16.)",
      "usage_table": "From line 5952-5968: Column Major memory + 16-bit element + Row Major VGPR layout → use GLOBAL_LOAD_TR16_B128 (or DS_LOAD_TR16_B128). The table can also be used when VGPR layout is column-major: simply reverse the 'memory order' meaning between Row and Column."
    },
    "key_line_references": {
      "section_10_9_wmma_load_transpose": "lines 5912-5968 — general description, instruction list, usage table",
      "section_11_2_4_lds_to_vgpr_matrix_load_transpose": "lines 6627-6645 — DS variant description",
      "wmma_matrix_storage_layouts": "lines 4445-4503 — VGPR lane/element layout tables for A, B, C/D matrices",
      "opcode_entries": "GLOBAL_LOAD_TR16_B128 = opcode 87 (line 8803, 27673); DS_LOAD_TR16_B128 = opcode 252 (line 25643)",
      "addressing_modes": "lines 5881-5895 — GV/GVS address formulas"
    }
  }
}