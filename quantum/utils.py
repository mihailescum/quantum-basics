def combine_basis_state(x, y, dim_x) -> int:
    if x >= (2 << (dim_x - 1)):
        raise ValueError("x has to be smaller than 2 ** dim_x!")

    return y << dim_x | x


def reduce_basis_state(a, dim_first) -> tuple[int, int]:
    x = a & ((2 << (dim_first - 1)) - 1)
    y = a >> dim_first
    return (x, y)
