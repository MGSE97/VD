#ifndef INTERSECTION_H_
#define INTERSECTION_H_

#include "ray.h"
#include "aabb.h"

bool RayBoxIntersectionMG( Ray & ray, AABB & box, float & t0, float & t1 );
bool RayBoxIntersection( const Ray & ray, AABB & box, float & t0, float & t1 );
bool BoxBoxIntersection( const AABB & box0, AABB & box1 );

#endif