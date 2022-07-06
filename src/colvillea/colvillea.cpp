#include <memory>
#include <filesystem>

#include <librender/device.h>
#include <librender/integrator.h>
#include <delegate/meshimporter.h>

using namespace colvillea;

int main(int argc, char* argv[])
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	auto dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
    std::vector<core::TriangleMesh> loadedMeshes = delegate::MeshImporter::loadMeshes(dir / "leftrightplane.obj");

	std::unique_ptr<core::Integrator> ptIntegrator = core::Integrator::createIntegrator(core::IntegratorType::WavefrontPathTracing);
	ptIntegrator->bindSceneTriangleMeshesData(loadedMeshes);
    ptIntegrator->render();

	//ptIntegrator->bindSceneTriangleMeshesData(delegate::MeshImporter::loadDefaultCube());
 //   ptIntegrator->render();
 //   ptIntegrator->render();
 //   ptIntegrator->render();
 //   ptIntegrator->render();
	return 0;
}