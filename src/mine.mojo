"""MINE characteristic-matrix kernel over caller-owned buffers."""

from std.algorithm import parallelize
from std.math import abs, log
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_ORIENTATION_WORK = 500_000


def zero_f64(values: FPtr, n: Int):
    var i = 0
    var zeros = SIMD[DType.float64, W](0.0)
    while i + W <= n:
        values.store(i, zeros)
        i += W
    while i < n:
        values[i] = 0.0
        i += 1


def zero_i64(values: IPtr, n: Int):
    var i = 0
    var zeros = SIMD[DType.int64, W](0)
    while i + W <= n:
        values.store(i, zeros)
        i += W
    while i < n:
        values[i] = 0
        i += 1


def equipartition_f64(values: FPtr, n: Int, bins: Int, mapping: IPtr) -> Int:
    var row_size = Float64(n) / Float64(bins)
    var i = 0
    var held = 0
    var current = 0
    while i < n:
        var size = 1
        while i + size < n and values[i] == values[i + size]:
            size += 1
        if held != 0 and abs(Float64(held + size) - row_size) >= abs(Float64(held) - row_size):
            current += 1
            held = 0
            row_size = Float64(n - i) / Float64(bins - current)
        for j in range(size):
            mapping[i + j] = Int64(current)
        i += size
        held += size
    return current + 1


def equipartition_i64(values: IPtr, n: Int, bins: Int, mapping: IPtr) -> Int:
    var row_size = Float64(n) / Float64(bins)
    var i = 0
    var held = 0
    var current = 0
    while i < n:
        var size = 1
        while i + size < n and values[i] == values[i + size]:
            size += 1
        if held != 0 and abs(Float64(held + size) - row_size) >= abs(Float64(held) - row_size):
            current += 1
            held = 0
            row_size = Float64(n - i) / Float64(bins - current)
        for j in range(size):
            mapping[i + j] = Int64(current)
        i += size
        held += size
    return current + 1


def clumps(dx: FPtr, qmap: IPtr, pmap: IPtr, n: Int, limit: Int) -> Int:
    var i = 0
    var count = 0
    var previous = Int64(-9223372036854775807)
    var marker = Int64(-1)
    while i < n:
        var size = 1
        var mixed = False
        while i + size < n and dx[i] == dx[i + size]:
            if qmap[i] != qmap[i + size]:
                mixed = True
            size += 1
        var label = qmap[i]
        if mixed:
            label = marker
            marker -= 1
        if label != previous:
            count += 1
            previous = label
        for j in range(size):
            pmap[i + j] = Int64(count - 1)
        i += size
    if count > limit:
        count = equipartition_i64(pmap, n, limit, pmap)
    return count


def hq(cum: IPtr, integer_log: FPtr, q: Int, p: Int, stride: Int, n: Int) -> Float64:
    var total = Float64(n)
    var total_log = integer_log[n]
    var accumulated = SIMD[DType.float64, W](0.0)
    var zeros = SIMD[DType.float64, W](0.0)
    var row = 0
    while row + W <= q:
        var values = (cum + row * stride + p - 1).strided_load[width=W](stride)
        var values_f64 = values.cast[DType.float64]()
        var logs = integer_log.gather(values.cast[DType.int]())
        accumulated += values.ne(0).select(
            (values_f64 / total) * (logs - total_log), zeros
        )
        row += W
    var result = -accumulated.reduce_add()[0]
    while row < q:
        var value = cum[row * stride + p - 1]
        if value != 0:
            var prob = Float64(value) / total
            result -= prob * (integer_log[Int(value)] - total_log)
        row += 1
    return result


def hp3(c: IPtr, integer_log: FPtr, s: Int, t: Int) -> Float64:
    if s == t:
        return 0.0
    var total = Float64(c[t - 1])
    var total_log = integer_log[Int(c[t - 1])]
    var result = 0.0
    var first = c[s - 1]
    if first != 0:
        var prob = Float64(first) / total
        result -= prob * (integer_log[Int(first)] - total_log)
    var second = c[t - 1] - first
    if second != 0:
        var prob = Float64(second) / total
        result -= prob * (integer_log[Int(second)] - total_log)
    return result


def hp3q(
    cum: IPtr,
    c: IPtr,
    integer_log: FPtr,
    q: Int,
    stride: Int,
    s: Int,
    t: Int,
) -> Float64:
    var total = Float64(c[t - 1])
    var total_log = integer_log[Int(c[t - 1])]
    var accumulated = SIMD[DType.float64, W](0.0)
    var zeros = SIMD[DType.float64, W](0.0)
    var row = 0
    while row + W <= q:
        var first = (cum + row * stride + s - 1).strided_load[width=W](stride)
        var first_f64 = first.cast[DType.float64]()
        var first_logs = integer_log.gather(first.cast[DType.int]())
        accumulated += first.ne(0).select(
            (first_f64 / total) * (first_logs - total_log), zeros
        )
        var second = (
            cum + row * stride + t - 1
        ).strided_load[width=W](stride) - first
        var second_f64 = second.cast[DType.float64]()
        var second_logs = integer_log.gather(second.cast[DType.int]())
        accumulated += second.ne(0).select(
            (second_f64 / total) * (second_logs - total_log), zeros
        )
        row += W
    var result = -accumulated.reduce_add()[0]
    while row < q:
        var first = cum[row * stride + s - 1]
        if first != 0:
            var prob = Float64(first) / total
            result -= prob * (integer_log[Int(first)] - total_log)
        var second = cum[row * stride + t - 1] - first
        if second != 0:
            var prob = Float64(second) / total
            result -= prob * (integer_log[Int(second)] - total_log)
        row += 1
    return result


def hp2q(
    cum: IPtr,
    c: IPtr,
    integer_log: FPtr,
    q: Int,
    stride: Int,
    s: Int,
    t: Int,
) -> Float64:
    if s == t:
        return 0.0
    var total = Float64(c[t - 1] - c[s - 1])
    var total_log = integer_log[Int(c[t - 1] - c[s - 1])]
    var accumulated = SIMD[DType.float64, W](0.0)
    var zeros = SIMD[DType.float64, W](0.0)
    var row = 0
    while row + W <= q:
        var values = (
            cum + row * stride + t - 1
        ).strided_load[width=W](stride) - (
            cum + row * stride + s - 1
        ).strided_load[width=W](stride)
        var values_f64 = values.cast[DType.float64]()
        var value_logs = integer_log.gather(values.cast[DType.int]())
        accumulated += values.ne(0).select(
            (values_f64 / total) * (value_logs - total_log), zeros
        )
        row += W
    var result = -accumulated.reduce_add()[0]
    while row < q:
        var value = cum[row * stride + t - 1] - cum[row * stride + s - 1]
        if value != 0:
            var prob = Float64(value) / total
            result -= prob * (integer_log[Int(value)] - total_log)
        row += 1
    return result


def optimize(
    qmap: IPtr,
    pmap: IPtr,
    n: Int,
    q: Int,
    p: Int,
    grid_x: Int,
    pstride: Int,
    xstride: Int,
    score: FPtr,
    c: IPtr,
    integer_log: FPtr,
    cum: IPtr,
    info: FPtr,
    hp: FPtr,
):
    if p == 1:
        zero_f64(score, grid_x - 1)
        return

    zero_i64(c, p)
    for i in range(n):
        c[Int(pmap[i])] += 1
    for i in range(1, p):
        c[i] += c[i - 1]

    zero_i64(cum, q * pstride)
    for i in range(n):
        cum[Int(qmap[i]) * pstride + Int(pmap[i])] += 1
    for row in range(q):
        for col in range(1, p):
            cum[row * pstride + col] += cum[row * pstride + col - 1]

    zero_f64(info, (p + 1) * xstride)
    for t in range(3, p + 1):
        for s in range(2, t + 1):
            hp[s * (pstride + 1) + t] = hp2q(
                cum, c, integer_log, q, pstride, s, t
            )

    var entropy_q = hq(cum, integer_log, q, p, pstride, n)

    for t in range(2, p + 1):
        var best = -1.7976931348623157e308
        for s in range(1, t + 1):
            var value = hp3(c, integer_log, s, t) - hp3q(
                cum, c, integer_log, q, pstride, s, t
            )
            if value > best:
                info[t * xstride + 2] = entropy_q + value
                best = value

    for cells in range(3, grid_x + 1):
        for t in range(cells, p + 1):
            var ct = Float64(c[t - 1])
            var best = -1.7976931348623157e308
            for s in range(cells - 1, t + 1):
                var cs = Float64(c[s - 1])
                var value = (
                    (cs / ct) * (info[s * xstride + cells - 1] - entropy_q)
                    - ((ct - cs) / ct) * hp[s * (pstride + 1) + t]
                )
                if value > best:
                    info[t * xstride + cells] = entropy_q + value
                    best = value

    for cells in range(p + 1, grid_x + 1):
        info[p * xstride + cells] = info[p * xstride + p]
    for cells in range(2, grid_x + 1):
        var denom = min(log(Float64(cells)), log(Float64(q)))
        score[cells - 2] = info[p * xstride + cells] / denom


def compute_orientation(
    partition_values: FPtr,
    clump_values: FPtr,
    partition_order: IPtr,
    clump_order: IPtr,
    n: Int,
    rows: Int,
    cols: Int,
    widths: IPtr,
    clump_factor: Float64,
    estimator: Int,
    score: FPtr,
    qtmp: IPtr,
    qmap: IPtr,
    pmap: IPtr,
    counts: IPtr,
    integer_log: FPtr,
    cum: IPtr,
    info: FPtr,
    hp: FPtr,
    pstride: Int,
    xstride: Int,
):
    for row in range(rows):
        var row_bins = row + 2
        var row_cols = Int(widths[row])
        var limit = max(Int(clump_factor * Float64(row_cols + 1)), 1)
        var q = equipartition_f64(partition_values, n, row_bins, qmap)
        for j in range(n):
            qtmp[Int(partition_order[j])] = qmap[j]
        for j in range(n):
            qmap[j] = qtmp[Int(clump_order[j])]
        var p = clumps(clump_values, qmap, pmap, n, limit)
        var grid_x = row_cols + 1
        if estimator == 1:
            grid_x = min(row_bins, grid_x)
        optimize(
            qmap, pmap, n, q, p, grid_x, pstride, xstride,
            score + row * cols, counts, integer_log, cum, info, hp,
        )


def compute_matrix(
    xx: FPtr,
    yy: FPtr,
    ix: IPtr,
    iy: IPtr,
    n: Int,
    rows: Int,
    cols: Int,
    widths: IPtr,
    clump_factor: Float64,
    estimator: Int,
    matrix: FPtr,
    transposed_score: FPtr,
    qtmp: IPtr,
    qmap: IPtr,
    pmap: IPtr,
    counts: IPtr,
    integer_log: FPtr,
    cum: IPtr,
    info: FPtr,
    hp: FPtr,
    pstride: Int,
    xstride: Int,
    qtmp2: IPtr,
    qmap2: IPtr,
    pmap2: IPtr,
    counts2: IPtr,
    cum2: IPtr,
    info2: FPtr,
    hp2: FPtr,
):
    zero_f64(matrix, rows * cols)

    def run_orientation(task: Int) {imm}:
        if task == 0:
            compute_orientation(
                yy, xx, iy, ix, n, rows, cols, widths,
                clump_factor, estimator, matrix, qtmp, qmap, pmap,
                counts, integer_log, cum, info, hp, pstride, xstride,
            )
        else:
            compute_orientation(
                xx, yy, ix, iy, n, rows, cols, widths,
                clump_factor, estimator, transposed_score,
                qtmp2, qmap2, pmap2, counts2, integer_log,
                cum2, info2, hp2, pstride, xstride,
            )

    if n * rows * cols >= PARALLEL_ORIENTATION_WORK:
        parallelize(run_orientation, 2, 2)
    else:
        run_orientation(0)
        run_orientation(1)

    for row in range(rows):
        var row_cols = Int(widths[row])
        var update = row_cols
        if estimator == 1:
            update = min(row + 1, row_cols)
        for col in range(update):
            if estimator == 0:
                matrix[col * cols + row] = max(
                    transposed_score[row * cols + col],
                    matrix[col * cols + row],
                )
            else:
                matrix[col * cols + row] = transposed_score[row * cols + col]


@export("mine_compute_matrix")
def mine_compute_matrix(
    xx_addr: Int,
    yy_addr: Int,
    ix_addr: Int,
    iy_addr: Int,
    n: Int,
    rows: Int,
    cols: Int,
    widths_addr: Int,
    clump_factor: Float64,
    estimator: Int,
    matrix_addr: Int,
    temporary_addr: Int,
    qtmp_addr: Int,
    qmap_addr: Int,
    pmap_addr: Int,
    counts_addr: Int,
    counts_log_addr: Int,
    cum_addr: Int,
    info_addr: Int,
    hp_addr: Int,
    pstride: Int,
    xstride: Int,
    qtmp2_addr: Int,
    qmap2_addr: Int,
    pmap2_addr: Int,
    counts2_addr: Int,
    cum2_addr: Int,
    info2_addr: Int,
    hp2_addr: Int,
) abi("C") -> Int:
    if (
        xx_addr == 0 or yy_addr == 0 or ix_addr == 0 or iy_addr == 0
        or widths_addr == 0
        or matrix_addr == 0 or temporary_addr == 0 or qtmp_addr == 0
        or qmap_addr == 0 or pmap_addr == 0 or counts_addr == 0
        or counts_log_addr == 0 or cum_addr == 0
        or info_addr == 0 or hp_addr == 0 or qtmp2_addr == 0
        or qmap2_addr == 0 or pmap2_addr == 0 or counts2_addr == 0
        or cum2_addr == 0 or info2_addr == 0 or hp2_addr == 0
    ):
        return 1
    if n < 2 or rows < 1 or cols < 1 or pstride < 1 or xstride < cols + 2:
        return 2
    if estimator < 0 or estimator > 1 or clump_factor <= 0.0:
        return 3
    compute_matrix(
        FPtr(unsafe_from_address=xx_addr),
        FPtr(unsafe_from_address=yy_addr),
        IPtr(unsafe_from_address=ix_addr),
        IPtr(unsafe_from_address=iy_addr),
        n,
        rows,
        cols,
        IPtr(unsafe_from_address=widths_addr),
        clump_factor,
        estimator,
        FPtr(unsafe_from_address=matrix_addr),
        FPtr(unsafe_from_address=temporary_addr),
        IPtr(unsafe_from_address=qtmp_addr),
        IPtr(unsafe_from_address=qmap_addr),
        IPtr(unsafe_from_address=pmap_addr),
        IPtr(unsafe_from_address=counts_addr),
        FPtr(unsafe_from_address=counts_log_addr),
        IPtr(unsafe_from_address=cum_addr),
        FPtr(unsafe_from_address=info_addr),
        FPtr(unsafe_from_address=hp_addr),
        pstride,
        xstride,
        IPtr(unsafe_from_address=qtmp2_addr),
        IPtr(unsafe_from_address=qmap2_addr),
        IPtr(unsafe_from_address=pmap2_addr),
        IPtr(unsafe_from_address=counts2_addr),
        IPtr(unsafe_from_address=cum2_addr),
        FPtr(unsafe_from_address=info2_addr),
        FPtr(unsafe_from_address=hp2_addr),
    )
    return 0
