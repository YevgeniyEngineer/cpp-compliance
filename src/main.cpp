#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

std::int32_t main(std::int32_t argc, const char *const argv[])
{
    struct LidarPoint final
    {
        constexpr LidarPoint() noexcept : x{0.0F}, y{0.0F}, z{0.0F}
        {
        }

        constexpr explicit LidarPoint(float x, float y, float z) noexcept : x{x}, y{y}, z{z}
        {
        }

        float x;
        float y;
        float z;
    };

    using LidarCloud = std::vector<LidarPoint>;

    LidarCloud lidar_cloud;

    static constexpr std::size_t NUMBER_OF_POINTS{100000U};

    lidar_cloud.reserve(NUMBER_OF_POINTS);

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> distribution{-100.0F, 100.0F};

    for (std::size_t i = 0U; i < NUMBER_OF_POINTS; ++i)
    {
        lidar_cloud.emplace_back(distribution(gen), distribution(gen), distribution(gen));
    }

    // Serialize data to const char*
    static constexpr std::size_t BUFFER_SAFETY_FACTOR{2U};
    const std::size_t blob_size = lidar_cloud.size() * sizeof(LidarPoint);
    const std::unique_ptr<char[]> buffer = std::make_unique<char[]>(
        BUFFER_SAFETY_FACTOR * (sizeof(std::size_t) + (NUMBER_OF_POINTS * sizeof(LidarPoint))));
    char *buffer_ptr = buffer.get();

    // Serialize the size of the vector
    const std::size_t size = lidar_cloud.size();
    std::memcpy(static_cast<void *>(buffer_ptr), static_cast<const void *>(&size), sizeof(size));
    buffer_ptr += sizeof(size);

    // Serialize the contents of the vector using iterators
    char *element_ptr = static_cast<char *>(static_cast<void *>(&lidar_cloud[0]));
    std::copy(element_ptr, element_ptr + (size * sizeof(LidarPoint)), buffer_ptr);

    // Deserialize data
    buffer_ptr = buffer.get();

    // Deserialize the size of the vector
    std::size_t deserialized_size;
    std::memcpy(static_cast<void *>(&deserialized_size), static_cast<const void *>(buffer_ptr),
                sizeof(deserialized_size));
    buffer_ptr += sizeof(deserialized_size);

    // Convert buffer_ptr to LidarPoint*
    LidarPoint *deserialized_data = static_cast<LidarPoint *>(static_cast<void *>(buffer_ptr));

    // Construct deserialized cloud from serialized data
    LidarCloud deserialized_cloud(deserialized_data, (deserialized_data + deserialized_size));

    // Compare with original version
    std::cout << "Original size: " << lidar_cloud.size() << " | Deserialized size: " << deserialized_size << std::endl;

    // Compare all elements
    const auto pointsEqual = [](const LidarPoint &p1, const LidarPoint &p2) {
        static constexpr float PRECISION = std::numeric_limits<float>::epsilon();

        return ((std::fabs(p1.x - p2.x) < PRECISION) && (std::fabs(p1.y - p2.y) < PRECISION) &&
                (std::fabs(p1.z - p2.z) < PRECISION));
    };

    bool clouds_equal = true;
    for (std::size_t point_number = 0U; point_number < lidar_cloud.size(); ++point_number)
    {
        if (!pointsEqual(lidar_cloud[point_number], deserialized_cloud[point_number]))
        {
            clouds_equal = false;
            break;
        }
    }

    std::cout << "Clouds equal: " << clouds_equal << std::endl;

    return 0;
}