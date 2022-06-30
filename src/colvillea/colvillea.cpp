#include <librender/device.h>

#include <memory>

int main(int argc, char* argv[])
{
    std::unique_ptr<Colvillea::Core::Device> optixDevice = Colvillea::Core::Device::createDevice(Colvillea::Core::DeviceType::OptiXDevice);

	return 0;
}