#ifndef _KARRAS_RANGE_
#define _KARRAS_RANGE_

// Karras 2012 range/split. delta() is shared by both so duplicate Morton
// codes stay a single tree: a midpoint shortcut on equal codes orphans leaves.
// `build_words` and OFF_KEYS_A / KEY_STRIDE must already be declared.

int clz32(uint x) {
    return x == 0u ? 32 : 31 - findMSB(x);
}

uint key_code(int i) {
    return build_words[OFF_KEYS_A + uint(i) * KEY_STRIDE];
}

int delta(int n, int i, int j) {
    if (j < 0 || j >= n) {
        return -1;
    }
    const uint ki = key_code(i);
    const uint kj = key_code(j);
    if (ki == kj) {
        return 32 + clz32(uint(i) ^ uint(j));
    }
    return clz32(ki ^ kj);
}

void determine_range(int n, int i, out int first, out int last) {
    const int d = delta(n, i, i + 1) >= delta(n, i, i - 1) ? 1 : -1;
    const int delta_min = delta(n, i, i - d);
    int l_max = 2;
    while (l_max <= n && delta(n, i, i + l_max * d) > delta_min) {
        l_max <<= 1;
    }
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (delta(n, i, i + (l + t) * d) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d;
    first = min(i, j);
    last = max(i, j);
}

int find_split(int n, int first, int last) {
    const int common_prefix = delta(n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int new_split = split + step;
        if (new_split < last && delta(n, first, new_split) > common_prefix) {
            split = new_split;
        }
    } while (step > 1);
    return split;
}

#endif
