#include <chrono>    // std::chrono
#include <cstddef>   // std::size_t
#include <cstring>   // std::memcpy
#include <exception> // std::terminate
#include <iostream>  // std::cout
#include <random>    // std::mt19937
#include <utility>   // std::index_sequence
#include <vector>    // std::vector

extern "C"
{
#include <emmintrin.h> // SSE2 intrinsics
}

namespace nostd
{
template <std::size_t... Indices>
inline static void copyBlock(char *&dest, const char *&src, std::index_sequence<Indices...>) noexcept
{
    ((dest[Indices] = src[Indices]), ...);
}

[[maybe_unused]] static inline void *memcpyUnrolled(void *dest, const void *src, std::size_t n_bytes) noexcept
{
    char *d = static_cast<char *>(dest);
    const char *s = static_cast<const char *>(src);

    static constexpr std::size_t BLOCK_SIZE = 4U;
    static constexpr std::size_t CONST = BLOCK_SIZE - 1U;

    const std::size_t unroll_count = n_bytes / BLOCK_SIZE;
    const std::size_t remaining_bytes = n_bytes & CONST;

    for (std::size_t i = 0U; i < unroll_count; ++i, s += BLOCK_SIZE, d += BLOCK_SIZE)
    {
        copyBlock(d, s, std::make_index_sequence<BLOCK_SIZE>{});
    }

    // Copy any remaining bytes
    for (std::size_t i = 0U; i < remaining_bytes; ++i)
    {
        d[i] = s[i];
    }

    return dest;
}

// [[maybe_unused]] static inline void *memcpySIMD(void *dest, const void *src, std::size_t n_bytes) noexcept
// {
//     char *d = static_cast<char *>(dest);
//     const char *s = static_cast<const char *>(src);

//     static constexpr std::size_t BLOCK_SIZE = 16U; // SSE2 can handle 16 bytes at a time
//     static constexpr std::size_t CONST = BLOCK_SIZE - 1U;

//     const std::size_t unroll_count = n_bytes / BLOCK_SIZE;
//     const std::size_t remaining_bytes = n_bytes & CONST;

//     for (std::size_t i = 0U; i < unroll_count; ++i, s += BLOCK_SIZE, d += BLOCK_SIZE)
//     {
//         __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s));
//         _mm_storeu_si128(reinterpret_cast<__m128i *>(d), data);
//     }

//     // Copy any remaining bytes
//     for (std::size_t i = 0U; i < remaining_bytes; ++i)
//     {
//         d[i] = s[i];
//     }

//     return dest;
// }

[[maybe_unused]] static inline void *memcpySIMD(void *const __restrict__ dest, const void *const __restrict__ src,
                                                std::size_t n_bytes) noexcept
{
    auto *const d = static_cast<std::uint8_t *const>(dest);
    const auto *const s = static_cast<const std::uint8_t *const>(src);

    static constexpr std::size_t BLOCK_SIZE = 16U; // SSE2 can handle 16 bytes at a time
    static constexpr std::size_t CONST = BLOCK_SIZE - 1U;

    const std::size_t unroll_count = n_bytes / (BLOCK_SIZE * 4); // Unroll by 4
    const std::size_t remaining_bytes = n_bytes & CONST;

    // Ensure proper alignment
    __m128i *d_aligned = reinterpret_cast<__m128i *>(d);
    const __m128i *s_aligned = reinterpret_cast<const __m128i *>(s);

    // Prefetch the data
    _mm_prefetch(reinterpret_cast<const std::uint8_t *const>(s_aligned), _MM_HINT_T0);

    // Unroll the loop by 4
    for (std::size_t i = 0U; i < unroll_count; ++i, s_aligned += 4, d_aligned += 4)
    {
        _mm_store_si128(d_aligned, _mm_load_si128(s_aligned));
        _mm_store_si128(d_aligned + 1, _mm_load_si128(s_aligned + 1));
        _mm_store_si128(d_aligned + 2, _mm_load_si128(s_aligned + 2));
        _mm_store_si128(d_aligned + 3, _mm_load_si128(s_aligned + 3));

        // Prefetch the next data
        _mm_prefetch(reinterpret_cast<const std::uint8_t *const>(s_aligned + 4), _MM_HINT_T0);
    }

    // Copy any remaining bytes
    for (std::size_t i = 0U; i < remaining_bytes; ++i)
    {
        d[i] = s[i];
    }

    return dest;
}

template <std::size_t... Indices>
inline static void simdCopy(char *&dest, const char *&src, std::index_sequence<Indices...>) noexcept
{
    // Unpack the index sequence and perform SIMD operations
    ((_mm_storeu_si128(reinterpret_cast<__m128i *>(dest + Indices * 16),
                       _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + Indices * 16)))),
     ...);
}

void memcpySIMDAndUnroll(void *dest, const void *src, std::size_t n_bytes)
{
    char *d = static_cast<char *>(dest);
    const char *s = static_cast<const char *>(src);

    static constexpr std::size_t SIMD_BLOCK_SIZE = 16U;
    static constexpr std::size_t UNROLL_FACTOR = 4;
    static constexpr std::size_t TOTAL_BLOCK_SIZE = SIMD_BLOCK_SIZE * UNROLL_FACTOR;

    const std::size_t blocks = n_bytes / TOTAL_BLOCK_SIZE;
    const std::size_t remaining_bytes = n_bytes % TOTAL_BLOCK_SIZE;

    for (std::size_t i = 0; i < blocks; ++i, s += TOTAL_BLOCK_SIZE, d += TOTAL_BLOCK_SIZE)
    {
        simdCopy(d, s, std::make_index_sequence<UNROLL_FACTOR>{});
    }

    // Handle any remaining bytes
    for (std::size_t i = 0; i < remaining_bytes; ++i)
    {
        d[i] = s[i];
    }
}

[[maybe_unused]] static inline void *memcpyUint64(void *const __restrict__ dest, const void *const __restrict__ src,
                                                  const std::size_t n_bytes) noexcept
{
    if ((nullptr != dest) && (nullptr != src))
    {
        auto *const d = static_cast<std::uint8_t *const>(dest);
        const auto *const s = static_cast<const std::uint8_t *const>(src);

        // Copy data in blocks of 64 bits (8 bytes)
        const std::size_t n_blocks = n_bytes / sizeof(std::uint64_t);
        const std::size_t n_bytes_total_in_blocks = n_blocks * sizeof(std::uint64_t);
        for (std::size_t i = 0; i < n_bytes_total_in_blocks; i += sizeof(std::uint64_t))
        {
            *reinterpret_cast<std::uint64_t *const>(&d[i]) = *reinterpret_cast<const std::uint64_t *const>(&s[i]);
        }

        // Copy any remaining bytes
        const std::size_t remaining_bytes = n_bytes & (sizeof(std::uint64_t) - 1U);
        for (std::size_t i = n_bytes_total_in_blocks; i < n_bytes; ++i)
        {
            d[i] = s[i];
        }
    }

    return dest;
}

template <typename DestinationType, typename SourceType, std::size_t BlockSize>
[[maybe_unused]] static inline void *memcpy(DestinationType *const __restrict__ dest,
                                            const SourceType *const __restrict__ src,
                                            const std::size_t n_bytes) noexcept
{
    // Validate BlockSize
    static_assert((BlockSize > 0) && ((BlockSize % 2) == 0) && (BlockSize <= sizeof(std::uint64_t)),
                  "BlockSize must be less or equal to 8 bytes, but strictly greater than 0");

    // Check for nullpointers
    if ((nullptr != dest) && (nullptr != src))
    {
        auto *const d = reinterpret_cast<std::uint8_t *const>(dest);
        const auto *const s = reinterpret_cast<const std::uint8_t *const>(src);

        // Check for buffer overlap
        if (((d >= s) && (d <= (s + n_bytes))) || ((s >= d) && (s <= (d + n_bytes))))
        {
            std::terminate();
        }

        // Copy data in blocks
        const std::size_t n_blocks = n_bytes / BlockSize;
        const std::size_t n_bytes_total_in_blocks = n_blocks * BlockSize;
        for (std::size_t i = 0U; i < n_bytes_total_in_blocks; i += BlockSize)
        {
            if constexpr (8U == BlockSize)
            {
                *reinterpret_cast<std::uint64_t *const>(&d[i]) = *reinterpret_cast<const std::uint64_t *const>(&s[i]);
            }
            else if constexpr (4U == BlockSize)
            {
                *reinterpret_cast<std::uint32_t *const>(&d[i]) = *reinterpret_cast<const std::uint32_t *const>(&s[i]);
            }
            else if constexpr (2U == BlockSize)
            {
                *reinterpret_cast<std::uint16_t *const>(&d[i]) = *reinterpret_cast<const std::uint16_t *const>(&s[i]);
            }
            else
            {
                // Handle the default case: one byte at a time
                d[i] = s[i];
            }
        }

        // Copy any remaining bytes
        for (std::size_t i = n_bytes_total_in_blocks; i < n_bytes; ++i)
        {
            d[i] = s[i];
        }
    }

    return dest;
}

[[maybe_unused]] static inline void *memcpySimple(void *const __restrict__ dest, const void *const __restrict__ src,
                                                  const std::size_t n_bytes) noexcept
{
    // Check for nullpointers
    if ((nullptr != dest) && (nullptr != src))
    {
        auto *const d = reinterpret_cast<std::uint8_t *const>(dest);
        const auto *const s = reinterpret_cast<const std::uint8_t *const>(src);

        // Check for buffer overlap
        if (((d >= s) && (d <= (s + n_bytes))) || ((s >= d) && (s <= (d + n_bytes))))
        {
            std::terminate();
        }

        // Copy bytes
        for (std::size_t i = 0U; i < n_bytes; ++i)
        {
            d[i] = s[i];
        }
    }

    return dest;
}

template <typename DestinationType, typename SourceType>
[[maybe_unused]] static inline void *memcpySimpleTemplated(DestinationType *const __restrict__ dest,
                                                           const SourceType *const __restrict__ src,
                                                           const std::size_t n_bytes) noexcept
{
    // Check for nullpointers
    if ((nullptr != dest) && (nullptr != src))
    {
        auto *const d = reinterpret_cast<std::uint8_t *const>(dest);
        const auto *const s = reinterpret_cast<const std::uint8_t *const>(src);

        // Check for buffer overlap
        if (((d >= s) && (d <= (s + n_bytes))) || ((s >= d) && (s <= (d + n_bytes))))
        {
            std::terminate();
        }

        // Copy bytes
        for (std::size_t i = 0U; i < n_bytes; ++i)
        {
            d[i] = s[i];
        }
    }

    return dest;
}

} // namespace nostd

int main()
{
    static constexpr std::size_t NUM_FLOATS = 40'000U;
    static constexpr std::size_t NUM_BYTES = NUM_FLOATS * sizeof(float);

    // Creating test data
    std::vector<char> src(NUM_BYTES);
    std::vector<float> dest_std(NUM_FLOATS);
    std::vector<float> dest_nostd(NUM_FLOATS);

    // Fill source with random byte data
    std::mt19937 generator{42};
    std::uniform_int_distribution<char> distribution;
    for (auto &byte : src)
    {
        byte = distribution(generator);
    }

    // Time std::memcpy
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        std::memcpy(dest_std.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "std::memcpy took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " microseconds." << std::endl;

        // Time nostd::memcpyUint64
        start = std::chrono::high_resolution_clock::now();
        nostd::memcpyUint64(dest_nostd.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "nostd::memcpyUint64 took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds."
                  << std::endl;

        // Time nostd::memcpyUnrolled
        start = std::chrono::high_resolution_clock::now();
        nostd::memcpyUnrolled(dest_nostd.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "nostd::memcpyUnrolled took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds."
                  << std::endl;

        // Time nostd::memcpySIMD
        start = std::chrono::high_resolution_clock::now();
        nostd::memcpySIMD(dest_nostd.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "nostd::memcpySIMD took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds."
                  << std::endl;

        // Time nostd::memcpySIMDAndUnroll
        start = std::chrono::high_resolution_clock::now();
        nostd::memcpySIMDAndUnroll(dest_nostd.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "nostd::memcpySIMDAndUnroll took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds."
                  << std::endl;

        // Time nostd::memcpy
        start = std::chrono::high_resolution_clock::now();
        nostd::memcpy<float, char, sizeof(float)>(dest_nostd.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "nostd::memcpy took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds."
                  << std::endl;

        start = std::chrono::high_resolution_clock::now();
        nostd::memcpySimpleTemplated(dest_nostd.data(), src.data(), NUM_BYTES);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "nostd::memcpySimple took: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds."
                  << std::endl;

        std::cout << std::endl;
    }

    return 0;
}