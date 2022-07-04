#include <librender/device.h>

#include <memory>

int main(int argc, char* argv[])
{
    std::unique_ptr<colvillea::core::Device> optixDevice = colvillea::core::Device::createDevice(colvillea::core::DeviceType::OptiXDevice);

	return 0;
}