//
//  raytracer.cl
//  OpenCL_Raytracer
//
//  Created by Stefan Matthes on 18.07.15.
//
//

////////////////////////////////////////////////////////////////////////////////

#ifndef RAYTRACER_KERNEL
#define RAYTRACER_KERNEL

#define SAMPLE_COUNT_AMBIENT 1
#define SAMPLE_COUNT_AMBIENT_FACTOR (1.f/SAMPLE_COUNT_AMBIENT)
#define SAMPLE_COUNT_DIRECT 2
#define SAMPLE_COUNT_DIRECT_FACTOR (1.f/SAMPLE_COUNT_DIRECT)
#define FAR_CLIPPING_PLANE 100.f
#define RAYTRACING_DEPTH 3
#define MAX_PENDING_QBVH_NODES 32


typedef struct QuadNode
{
    float4 bbox[2*3];
    unsigned int child_nodes[4];
    unsigned int number_of_spheres[4];
}QuadNode;

typedef struct Sphere
{
    float4 pos_and_r;
    float4 color;
    unsigned int texture_index;
    float reflection;
    float refraction;
    float eta;
}Sphere;

////////////////////////////////////////////////////////////////////////////////

float getRandomPixelOffset(unsigned int *seed);

float4 getRandomVector(unsigned int *seed);

float4 reflect(const float4 incident_ray, const float4 surface_normal);

float4 refract(const float4 incident_ray, const float4 surface_normal, const float eta);

float4 refractThroughSphere(const float4 incident_ray, const float4 surface_normal, float4* intersection_point, const float4 sphere_pos, const float radius, const float eta);

bool raySphereIntersection(const Sphere sphere, const float4 ray_origin, const float4 ray_direction, const float max_distance, float* distance);

unsigned int rayQBoxIntersection(const float4 ray_origin, const float4 ray_direction, const float4* bbox, float max_distance, unsigned int* child_nodes);

bool lightOcclusion(const __global Sphere* sphere_buffer, const unsigned int sphere_count, const unsigned int self, const float4 point, const float4 light_direction, float light_distance);

float4 lightIntensity(const __global Sphere* sphere_buffer, const unsigned int sphere_count, const unsigned int self, const float4 point, const float4 normal, const Sphere light, unsigned int *seed, const float ambient_lighting);

float ambientLighting(const __global Sphere* sphere_buffer, const unsigned int sphere_count, const unsigned int self, const float4 point, const float4 normal, unsigned int seed);

float4 directIllumination(const __global Sphere* sphere_buffer, const unsigned int sphere_count, const unsigned int first_light_index, const unsigned int self, const float4 point, const float4 normal, unsigned int* seed);

unsigned int trace(const __global QuadNode* qbvh_buffer, const __global Sphere* sphere_buffer, const unsigned int sphere_count, const float4 ray_origin, const float4 ray_direction, unsigned int *index, float *distance);

float4 rayDirection(const __global float* cam_pose, const float x, const float y);

////////////////////////////////////////////////////////////////////////////////

float getRandomPixelOffset(unsigned int *seed)
{
    const unsigned int a = 48171;
    const unsigned int m = 1<<16;
    const float scaling_factor = 1.0f/float(1<<16 - 1);
    float offset = float(*seed)*scaling_factor - 0.5f;
    //const float scaling_factor = 1.0f/float(1<<16 - 1);
    //float offset = float(*seed)*scaling_factor - 0.5f;
    
    *seed = (a * *seed) % m;
    return offset;
}


float4 getRandomVector(unsigned int *seed)
{
    const unsigned int a = 48171;
    const unsigned int m = 1<<16;
    const float scaling_factor = 2.f/float(1<<16 - 1);
    
    float4 random_ray;
    
    random_ray.x = float(*seed);
    *seed = (a * *seed) % m;
    random_ray.y = float(*seed);
    *seed = (a * *seed) % m;
    random_ray.z = float(*seed);
    *seed = (a * *seed) % m;
    
    random_ray = random_ray*scaling_factor - 1.f;
    random_ray.w = 0.f;
    
    return normalize(random_ray);
}

float4 reflect(const float4 incident_ray, const float4 surface_normal)
{
    return incident_ray - 2.f*dot(surface_normal, incident_ray) * surface_normal;
}

float4 refract(const float4 incident_ray, const float4 surface_normal, const float eta)
{
    float4 refracted_ray;
    
    float n_dot_i = dot(surface_normal, incident_ray);
    float lambda = 1.f - eta * eta * (1.f - n_dot_i * n_dot_i);
    
    if(lambda >= 0.f)
    refracted_ray = eta * incident_ray - surface_normal * (eta * n_dot_i + sqrt(lambda));
    
    refracted_ray.w = 0.f;
    return refracted_ray;
}

float4 refractThroughSphere(const float4 incident_ray,
                            const float4 surface_normal,
                            float4* intersection_point,
                            const float4 sphere_pos,
                            const float radius,
                            const float eta)
{
    float4 refracted_ray = refract(incident_ray, surface_normal, 1.f/eta);
    
    float4 L = sphere_pos - *intersection_point;
    L.w = 0.f;
    float LdotRR = dot(L, refracted_ray);
    float D2 = dot(L, L) - LdotRR * LdotRR;
    float Distance = LdotRR + sqrt(radius * radius - D2);
    *intersection_point += refracted_ray * Distance;
    
    float4 NewNormal = (sphere_pos - *intersection_point) / radius;
    float4 refracted_ray2 = refract(refracted_ray, NewNormal, eta);
    
    return refracted_ray2;
}

bool raySphereIntersection(const Sphere sphere,
                           const float4 ray_origin,
                           const float4 ray_direction,
                           const float max_distance,
                           float* distance)
{
    float4 sphere_direction = sphere.pos_and_r - ray_origin;
    sphere_direction.w = 0.f;
    
    float lambda = dot(sphere_direction, ray_direction);
    
    if(lambda > 0.f)//sphere is in front of camera
    {
        float distance_squared = dot(sphere_direction,sphere_direction) - lambda*lambda;
        float radius_squared = sphere.pos_and_r.w * sphere.pos_and_r.w;
        if(distance_squared < radius_squared)
        {
            *distance = lambda - sqrt(radius_squared - distance_squared);
            if(*distance >= 0.f && (*distance < max_distance))
            return true;
        }
    }
    return false;
}

unsigned int rayQBoxIntersection(const float4 ray_origin, const float4 ray_direction, const float4* bbox, float max_distance, unsigned int* child_nodes)
{
    float4 dirfrac = 1.f/ray_direction;
    
    float4 t1 = (bbox[0] - ray_origin.x)*dirfrac.x;
    float4 t2 = (bbox[1] - ray_origin.x)*dirfrac.x;
    float4 t3 = (bbox[2] - ray_origin.y)*dirfrac.y;
    float4 t4 = (bbox[3] - ray_origin.y)*dirfrac.y;
    float4 t5 = (bbox[4] - ray_origin.z)*dirfrac.z;
    float4 t6 = (bbox[5] - ray_origin.z)*dirfrac.z;
    
    float4 tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
    float4 tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));
    
    int4 comp = isgreater(tmax, 0.f) & isgreater(tmax, tmin) & isless(tmin, max_distance);
    unsigned int intersection_count = 0;
    
    if( comp.x ) child_nodes[intersection_count++] = 0;
    if( comp.y ) child_nodes[intersection_count++] = 1;
    if( comp.z ) child_nodes[intersection_count++] = 2;
    if( comp.w ) child_nodes[intersection_count++] = 3;
    
    return intersection_count;
}

bool lightOcclusion(const __global Sphere* sphere_buffer,
                    const unsigned int sphere_count,
                    const unsigned int self,
                    const float4 point,
                    const float4 light_direction,
                    float light_distance)
{
    float distance;
    for(unsigned int i=0; i<sphere_count; i++)
    {
        if(i == self) continue;
        if( raySphereIntersection(sphere_buffer[i], point, light_direction, light_distance, &distance) )
        return true;
    }
    return false;
}

float4 lightIntensity(const __global Sphere* sphere_buffer,
                      const unsigned int sphere_count,
                      const unsigned int self,
                      const float4 point,
                      const float4 normal,
                      const Sphere light,
                      unsigned int *seed,
                      const float ambient_lighting)
{
    float4 light_position = light.pos_and_r + getRandomVector(seed)*light.pos_and_r.w;
    float4 light_direction = light_position - point;
    light_direction.w = 0.f;
    float light_distance_squared = dot(light_direction, light_direction);
    float light_distance = sqrt(light_distance_squared);
    light_direction /= light_distance;
    
    float attenuation = 0.1f + 0.f*light_distance + 0.1f * light_distance_squared;
    
    float n_dot_ld = dot(normal, light_direction);
    if(n_dot_ld > 0.f)
    {
        if( lightOcclusion(sphere_buffer, sphere_count, self, point, light_direction, light_distance) == false)
            return light.color * (ambient_lighting*light.color.w + n_dot_ld) / attenuation;
    }
    
    return light.color * ambient_lighting*light.color.w / attenuation;
}


float ambientLighting(const __global Sphere* sphere_buffer, const unsigned int sphere_count, const unsigned int self, const float4 point, const float4 normal, unsigned int seed)
{
    float ambient_lighting = 0.f;
    
    for(int i = 0; i < SAMPLE_COUNT_AMBIENT; i++)
    {
        float4 random_ray = getRandomVector(&seed);
        float n_dot_rr = dot(normal, random_ray);
        
        if(n_dot_rr < 0.f)
        {
            random_ray = -random_ray;
            n_dot_rr = -n_dot_rr;
        }
        
        float object_distance = 100.f;
        
        float distance;
        for(unsigned int i=0; i<sphere_count; i++)
        {
            if(i == self) continue;
            
            if( raySphereIntersection(sphere_buffer[i], point, random_ray, object_distance, &distance) )
            object_distance = distance;
        }
        
        ambient_lighting += n_dot_rr / (1.f + object_distance*object_distance);
    }
    
    return 1.f - ambient_lighting * SAMPLE_COUNT_AMBIENT_FACTOR;
}


float4 directIllumination(const __global Sphere* sphere_buffer,
                          const unsigned int sphere_count,
                          const unsigned int first_light_index,
                          const unsigned int self,
                          const float4 point,
                          const float4 normal,
                          unsigned int* seed)
{
    float4 lights_intensity = float4(0.f);
    float ambient_lighting = ambientLighting(sphere_buffer, first_light_index, self, point, normal, *seed);
    
    for(unsigned int light_index = first_light_index; light_index<sphere_count; light_index++)
    {
        for(int i=0; i<SAMPLE_COUNT_DIRECT; i++)
        {
            lights_intensity += lightIntensity(sphere_buffer, first_light_index, self, point, normal, sphere_buffer[light_index], seed, ambient_lighting);
        }
    }
    
    return lights_intensity * SAMPLE_COUNT_DIRECT_FACTOR;
}


unsigned int trace(const __global QuadNode* qbvh_buffer,
                   const __global Sphere* sphere_buffer,
                   const unsigned int sphere_count,
                   const float4 ray_origin,
                   const float4 ray_direction,
                   unsigned int *idx,
                   float *max_distance)
{
    unsigned int pending_child_nodes[MAX_PENDING_QBVH_NODES] = {};
    unsigned int child_nodes[4]                              = {};
    unsigned int cur_node                                    = 0;
    unsigned int next_node                                   = 0;
    unsigned int new_intersections                           = 0;
    unsigned int intersection_count                          = 1;
    bool hit                                                 = false;
    
    while( cur_node < intersection_count )
    {
        QuadNode qbvh = qbvh_buffer[pending_child_nodes[cur_node]];
        new_intersections = rayQBoxIntersection(ray_origin, ray_direction, qbvh.bbox, *max_distance, child_nodes);
        
        for(unsigned int i=0; i<new_intersections; i++)
        {
            next_node = child_nodes[i];
            if(qbvh.number_of_spheres[next_node] > 0) //test all spheres in this qbvh node
            {
                unsigned int first_sphere_index = qbvh.child_nodes[next_node];
                float distance;
                for(unsigned int i=first_sphere_index; i<first_sphere_index+qbvh.number_of_spheres[next_node]; i++)
                {
                    if( raySphereIntersection(sphere_buffer[i], ray_origin, ray_direction, *max_distance, &distance) )
                    {
                        hit = true;
                        *idx = i;
                        *max_distance = distance;
                    }
                }
            }
            else //add new nodes to check
            {
                pending_child_nodes[intersection_count++] = qbvh.child_nodes[next_node];
            }
        }
        cur_node++;
    }
    
    return hit;
}

float4 rayDirection(const __global float* cam_pose, const float x, const float y)
{
    //Gnomonical camera model
    /*float focal_length_inv = 0.002f;
     float cx_inv = focal_length_inv*0.5f*(get_global_size(0)-1.f);
     float cy_inv = focal_length_inv*0.5f*(get_global_size(1)-1.f);
     
     float4 ray;
     ray.x = focal_length_inv*x - cx_inv;
     ray.y = focal_length_inv*y - cy_inv;
     ray.z = 1.f;
     ray.w = 0.f;*/
    
    //Stereographic camera model
    float focal_length_inv = 0.002f;
    float x_c = x - 0.5f*(get_global_size(0)-1.f);
    float y_c = y - 0.5f*(get_global_size(1)-1.f);
    float radius = sqrt(x_c*x_c + y_c*y_c);
    float theta = 2.f*atan(radius*0.5f*focal_length_inv);
    float ratio = tan(theta)/radius;
    
    float4 ray = (float4)(ratio * x_c, ratio * y_c, 1.f, 0.f);
    float4 unit_ray = normalize(ray);
    float4 ray_direction = (float4)(cam_pose[0]*unit_ray.x+cam_pose[1]*unit_ray.y+cam_pose[2]*unit_ray.z,
                                    cam_pose[4]*unit_ray.x+cam_pose[5]*unit_ray.y+cam_pose[6]*unit_ray.z,
                                    cam_pose[8]*unit_ray.x+cam_pose[9]*unit_ray.y+cam_pose[10]*unit_ray.z,
                                    0.f);
    
    return ray_direction;
}

__kernel void raytracer(const __global QuadNode* qbvh_buffer,
                        const __global Sphere* sphere_buffer,
                        const __global unsigned int* sphere_count_buffer,
                        const __global unsigned int* first_light_index_buffer,
                        const __global float* cam_pose,
                        const __global uint* seed_buffer,
                        __write_only image2d_t output)
{
    const unsigned int global_x  = get_global_id(0);
    const unsigned int global_y  = get_global_id(1);
    const unsigned int global_id = global_y*get_global_size(0) + global_x;
    
    unsigned int seed = seed_buffer[global_id];
    
    float gx = (float)global_x;
    float gy = (float)global_y;
    float4 total_color = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for(float dx=-0.3f; dx<=0.3f; dx+=0.3f)//super sampling (9 rays per pixel)
    {
        for(float dy=-0.3f; dy<=0.3f; dy+=0.3f)
        {
            float4 ray_origin = (float4)(cam_pose[3], cam_pose[7], cam_pose[11], 0.f);
            float4 ray_direction = rayDirection(cam_pose, gx+dx, gy+dy);
        
            float4 color = (float4)(0.f, 0.f, 0.f, 0.0f);
    
            float distance = FAR_CLIPPING_PLANE;
            unsigned int sphere_idx;
            const unsigned int sphere_count = *sphere_count_buffer;
            const unsigned int first_light_index = *first_light_index_buffer;
            
            float factor = 1.f;
            for(int iter=0; iter<RAYTRACING_DEPTH; iter++)
            {
                if( trace(qbvh_buffer, sphere_buffer, sphere_count, ray_origin, ray_direction, &sphere_idx, &distance) )
                {
                    Sphere sphere = sphere_buffer[sphere_idx];
                    if( sphere_idx < first_light_index )
                    {
                        ray_origin += ray_direction*distance;
                        float4 normal = (ray_origin - sphere.pos_and_r)/sphere.pos_and_r.w;
                        normal.w = 0.f;
            
                        float opaque_factor = 1.f - sphere.reflection - sphere.refraction;
                        if( opaque_factor > 0.f )
                        {
                            color += factor*opaque_factor*sphere.color*directIllumination(sphere_buffer, sphere_count, first_light_index, sphere_idx, ray_origin, normal, &seed);
                        }
                        factor *= sphere.reflection;
                        if(factor > 0.f)
                        {
                            distance = FAR_CLIPPING_PLANE;
                            ray_direction = reflect(ray_direction, normal);
                        }
                        else
                            break;
                    }
                    else
                        color += factor*sphere.color; //light color
                }
                else
                    color += factor*(float4)(0.1f, 0.1f, 0.1f, 0.0f); //background color
            }
            
            total_color += color; //super sampling
        }
    }
    total_color/=9.f;
    
    write_imagef(output, (int2)(global_x,global_y), total_color);
}

#endif
