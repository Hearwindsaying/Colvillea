#pragma once

namespace colvillea
{
namespace core
{
// TODO: tmps
struct Camera
{
    vec3f m_camera_pos;
    vec3f m_camera_d00;
    vec3f m_camera_ddu;
    vec3f m_camera_ddv;
};
} // namespace core
} // namespace colvillea