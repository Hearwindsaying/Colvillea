#include <memory>
#include <filesystem>

#include <librender/device.h>
#include <delegate/meshimporter.h>

using namespace colvillea;


int main(int argc, char* argv[])
{
    std::unique_ptr<colvillea::core::Device> optixDevice = colvillea::core::Device::createDevice(colvillea::core::DeviceType::OptiXDevice);

	auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
    std::vector<core::TriangleMesh> loadedMeshes = delegate::MeshImporter::loadMeshes(dir / "leftrightplane.obj");

	return 0;
}