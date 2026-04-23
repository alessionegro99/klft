#!/usr/bin/env python3
"""Mean and blocked error for KLFT temporal Wilson-loop output."""

import argparse
import math
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a KLFT temporal Wilson-loop file and print mean and blocked "
            "error for each (L, T). Rows with step < THERM are discarded."
        )
    )
    parser.add_argument("file", help="temporal Wilson-loop output file")
    parser.add_argument("THERM", type=int, help="thermalization step cutoff")
    parser.add_argument("BLOCK", type=int, help="blocking size in measurements")
    return parser.parse_args()


def parse_loop_size(token, filename, line_number, name):
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(
            f"{filename}:{line_number}: could not parse {name}"
        ) from exc

    integer_value = int(value)
    if value != integer_value:
        raise ValueError(f"{filename}:{line_number}: {name} is not an integer")
    return integer_value


def read_wtemp_values(filename, therm):
    values_by_loop = {}
    with open(filename, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split()
            if len(fields) < 4:
                raise ValueError(
                    f"{filename}:{line_number}: expected at least 4 columns"
                )

            try:
                step = int(fields[0])
                wilson_loop = float(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"{filename}:{line_number}: could not parse step/W_temp"
                ) from exc

            if step < therm:
                continue

            loop_size = (
                parse_loop_size(fields[1], filename, line_number, "L"),
                parse_loop_size(fields[2], filename, line_number, "T"),
            )
            values_by_loop.setdefault(loop_size, []).append(wilson_loop)

    return values_by_loop


def mean(values):
    return sum(values) / len(values)


def blocked_error(values, block_size):
    nblocks = len(values) // block_size
    if nblocks < 2:
        return float("nan"), nblocks

    block_means = []
    for block_index in range(nblocks):
        start = block_index * block_size
        block = values[start : start + block_size]
        block_means.append(mean(block))

    block_mean = mean(block_means)
    variance = sum((value - block_mean) ** 2 for value in block_means)
    variance /= nblocks - 1
    return math.sqrt(variance / nblocks), nblocks


def main():
    args = parse_args()
    if args.THERM < 0:
        sys.exit("THERM must be non-negative")
    if args.BLOCK < 1:
        sys.exit("BLOCK must be >= 1")

    try:
        values_by_loop = read_wtemp_values(args.file, args.THERM)
    except OSError as exc:
        sys.exit(f"failed to read {args.file}: {exc}")
    except ValueError as exc:
        sys.exit(str(exc))

    if not values_by_loop:
        sys.exit("no temporal Wilson-loop measurements remain after thermalization cut")

    print("# L T n_measurements block_size n_blocks tail mean error")
    for loop_size in sorted(values_by_loop):
        values = values_by_loop[loop_size]
        value_mean = mean(values)
        error, nblocks = blocked_error(values, args.BLOCK)
        tail = len(values) - nblocks * args.BLOCK
        print(
            f"{loop_size[0]} {loop_size[1]} "
            f"{len(values)} {args.BLOCK} {nblocks} {tail} "
            f"{value_mean:.12g} {error:.12g}"
        )


if __name__ == "__main__":
    main()
