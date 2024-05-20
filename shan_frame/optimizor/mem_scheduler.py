from numpy import float64
import matplotlib
from ..ir import Model


class _Rect:
    idx: float64
    height: int
    weight: int
    start: int
    addr: int

    def __init__(self, idx: float64, height: int, weight: int, start: int) -> None:
        self.idx = idx
        self.height = height
        self.weight = weight
        self.start = start


# block: (start, end)
Blocks = list[tuple[float, float]]


class MemoryFootprint:
    # a column contains a list of allocated blocks with (start, end)
    columns: list[Blocks]

    def __init__(self, op_num: int) -> None:
        self.columns = [[] for _ in range(op_num)]


class MemoryScheduler:
    def _get_rect(self, model: Model) -> list[_Rect]:
        rect_list = []
        op_idx_list = list(model.operators.keys())
        op_idx_list.sort()
        assert op_idx_list[0] == 0 and op_idx_list[-1] == len(
            op_idx_list) - 1, "input model is not trimmed"
        for op_idx in op_idx_list:
            op = model.operators[op_idx]
            tensor_idx = op.output_idx
            tensor = model.tensors[tensor_idx]
            tensor_size = tensor.dim_n * tensor.dim_h * tensor.dim_w * tensor.dim_c
            tensor_lifetime = max(tensor.dst_op) - op_idx
            rect = _Rect(tensor_idx, tensor_size, tensor_lifetime, op_idx)
            rect_list.append(rect)
        return rect_list

    def schedule(self, model: Model):
        rect_list = self._get_rect(model)
        rect_list.sort(key=lambda rect: rect.height, reverse=True)
        footprint = MemoryFootprint(len(model.operators))
        for rect in rect_list:
            # find all free blocks (slots)
            takeup_columns = [rect.start + col for col in range(rect.weight)]
            total_slots: Blocks = [(0, float('inf'))]
            for col_idx in takeup_columns:
                current_slots: Blocks = []
                col = footprint.columns[col_idx]
                last_end = 0
                for start, end in col:
                    free_size = last_end - start
                    if free_size >= rect.height:
                        current_slots.append((last_end, start))
                    last_end = end
                current_slots.append((last_end, float('inf')))
                total_slots = inter_slots(total_slots, current_slots)
            total_slots.sort(key=lambda slot: slot[1]-slot[0])
            addr_head = total_slots[0][0]
            addr_tail = addr_head + rect.height
            model.tensors[rect.idx].addr = int(addr_head)
            rect.addr = int(addr_head)
            # mark memory block as taken
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
        raise NotImplementedError()


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
            start, end = min(start1, start2), max(end1, end2)
            inter_slots.append((start, end))
    return inter_slots
