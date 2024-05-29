from numpy import float64
import matplotlib

from ..ir.operator import Conv2D, DepthConv2D
from ..ir import Model
from ..utils import Rect, get_align_groups, get_buf_rect, get_rect


# block: (start, end)
Blocks = list[tuple[float, float]]


class MemoryFootprint:
    # a column contains a list of allocated blocks with (start, end)
    columns: list[Blocks]

    def __init__(self, op_num: int) -> None:
        self.columns = [[] for _ in range(op_num)]

    def __str__(self) -> str:
        result = ""
        for idx, col in enumerate(self.columns):
            result += f"{idx}: "
            for block in col:
                result += f"({(int(block[0]), int(block[1]))})"
            result += "\n"
        return result

class MemoryScheduler:
    def place_rect(self, rect: Rect, footprint: MemoryFootprint):
        takeup_columns = [rect.start + col for col in range(rect.width)]
        addr_head = rect.addr
        addr_tail = addr_head + rect.height
        # mark memory block as taken
        takeup_columns = [rect.start + col for col in range(rect.width)]
        for col_idx in takeup_columns:
            col = footprint.columns[col_idx]
            new_col_head = [
                block for block in col if block[1] <= addr_head]
            new_col_tail = [
                block for block in col if block[0] >= addr_tail]
            new_block = (addr_head, addr_tail)

            if len(new_col_head) != 0 and new_col_head[-1][1] == addr_head:
                prev_block = new_col_head.pop(-1)
                new_block = (prev_block[0], addr_tail)
            if len(new_col_tail) != 0 and new_col_tail[0][0] == addr_tail:
                next_block = new_col_tail.pop(0)
                new_block = (new_block[0], next_block[1])

            new_col = new_col_head
            new_col.append(new_block)
            new_col.extend(new_col_tail)
            footprint.columns[col_idx] = new_col
    
    def find_slots(self, rect: Rect, footprint: MemoryFootprint) -> Blocks:
        takeup_columns = [rect.start + col for col in range(rect.width)]
        # find all free blocks (slots)
        total_slots: Blocks = [(0, float('inf'))]
        for col_idx in takeup_columns:
            current_slots: Blocks = []
            col = footprint.columns[col_idx]
            last_end = 0
            for start, end in col:
                free_size = start - last_end
                if free_size >= rect.height:
                    current_slots.append((last_end, start))
                last_end = end
            current_slots.append((last_end, float('inf')))
            # print(f"col {col_idx} slots: {current_slots}")
            total_slots = inter_slots(total_slots, current_slots)
            # print(f"total slots after inter: {total_slots}")
        total_slots = list(
            filter(lambda slot: slot[1]-slot[0] >= rect.height, total_slots))
        return total_slots

    def schedule(self, model: Model) -> int:
        rect_list = get_rect(model)
        rect_list.sort(key=lambda rect: rect.width * rect.height, reverse=True)
        footprint = MemoryFootprint(len(model.operators))
        align_groups = get_align_groups(model)
        # fit op activations
        for rect in rect_list:
            # TODO: Add cache feature for optimization
            total_slots = self.find_slots(rect, footprint)
            
            # check if there is alignment requirement
            align_group_idx = -1
            for idx, align_group in enumerate(align_groups):
                if rect.idx in align_group[0]:
                    align_base = align_group[2]
                    align_step = align_group[1]
                    align_group_idx = idx
                    if align_base >= 0:
                        # there is an alignment base, pad all slots
                        for slot_idx, slot in enumerate(total_slots):
                            base_diff = abs(slot[0] - align_base)
                            pad_size = align_step - base_diff % align_step
                            pad_size = 0 if pad_size == align_step else pad_size
                            total_slots[slot_idx] = (slot[0] + pad_size, slot[1])
                        # remove slots smaller than rect after alignment
                        total_slots = list(
                            filter(lambda slot: slot[1]-slot[0] >= rect.height, total_slots))
            total_slots.sort(key=lambda slot: slot[1]-slot[0])
            rect.addr = int(total_slots[0][0])
            self.place_rect(rect, footprint)
            model.tensors[rect.idx].addr = rect.addr
            if align_group_idx >= 0 and align_groups[align_group_idx][2] < 0:
                # need to update align base of the align group
                align_group = align_groups[align_group_idx]
                align_step = align_group[1]
                align_rects = align_group[0]
                align_groups[align_group_idx] = (align_rects, align_step, rect.addr)
                
        # find peak memory usage
        peak_mem = int(max(column[-1][1] if len(column) > 0
                       else 0 for column in footprint.columns))
        # fit minimum buffer for ops
        buf_rect_list = get_buf_rect(model)
        for buf_rect in buf_rect_list:
            op = model.operators[int(buf_rect.idx)]
            assert isinstance(op, Conv2D) or isinstance(op, DepthConv2D), "only (depthwise) conv2d requires buffer"
            
            total_slots = self.find_slots(buf_rect, footprint)
            total_slots.sort(key=lambda slot: slot[1]-slot[0])
            # buffers do not interfere with each other
            # thus no need to really place them on the footprint
            # only the address and peak memory matter
            if total_slots[0][0] + buf_rect.height > peak_mem:
                # peak mem is raised due to buffer requirement
                peak_mem = int(total_slots[0][0] + buf_rect.height)
                buf_rect.addr = int(total_slots[0][0])
                op.buffer_addr = buf_rect.addr
                op.buffer_size = buf_rect.height
            else:
                # find largest slot that does not raise peak mem
                total_slots.sort(key=lambda slot: slot[1]-slot[0] if slot[1]!=float("inf") else peak_mem - slot[0], reverse=True)
                slot = total_slots[0]
                buf_rect.addr = int(slot[0])
                op.buffer_addr = buf_rect.addr
                op.buffer_size = int(slot[1]-slot[0] if slot[1]!=float("inf") else peak_mem - slot[0])
        return peak_mem


def inter_slots(slots1: Blocks, slots2: Blocks) -> Blocks:
    slots2_skip = 0
    inter_slots = []
    for start1, end1 in slots1:
        for start2, end2 in slots2[slots2_skip:]:
            if end1 <= start2:
                break
            if end2 <= start1:
                slots2_skip += 1
                continue
            # slot1 and slot2 has intersection
            start, end = max(start1, start2), min(end1, end2)
            inter_slots.append((start, end))
    return inter_slots
