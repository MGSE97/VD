#include "stdafx.h"

#include "intersection.h"

//http://ompf.org/ray/ray_box.html
//http://www.flipcode.com/archives/SSE_RayBox_Intersection_Test.shtml
//http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
bool RayBoxIntersectionMG( Ray & ray, AABB & box, float & t0, float & t1 )
{
	const Vector3 r_dir = Vector3( 1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z ); // TODO pøesunout pøímo do Ray

	float box_t0 = ( box.lower_bound().x - ray.origin.x ) * r_dir.x;
	float box_t1 = ( box.upper_bound().x - ray.origin.x ) * r_dir.x;
	float t_min = MIN( box_t0, box_t1 );
	float t_max = MAX( box_t0, box_t1 );

	box_t0 = ( box.lower_bound().y - ray.origin.y ) * r_dir.y;
	box_t1 = ( box.upper_bound().y - ray.origin.y ) * r_dir.y;
	t_min = MAX( MIN( box_t0, box_t1 ), t_min );
	t_max = MIN( MAX( box_t0, box_t1 ), t_max );
			
	box_t0 = ( box.lower_bound().z - ray.origin.z ) * r_dir.z;
	box_t1 = ( box.upper_bound().z - ray.origin.z ) * r_dir.z;
	t_min = MAX( MIN( box_t0, box_t1 ), t_min );
	t_max = MIN( MAX( box_t0, box_t1 ), t_max );
	
	if ( ( t_max >= 0 ) && ( t_max >= t_min ) &&
		( t0 <= t_max ) && ( t1 >= t_min ) )
	{
		// ray protíná box nìkde mezi t0 až t1
		t0 = MAX( t_min, t0 );
		t1 = MIN( t_max, t1 );

		return true;
	}

	return false;
}

//http://jcgt.org/published/0002/02/02/paper.pdf
bool RayBoxIntersection( const Ray & ray, AABB & box, float & t0, float & t1 )
{	
	float t_x_min = ( box[ray.direction_sign[0]].x - ray.origin.x ) * ray.inv_direction.x;
	float t_x_max = ( box[1 - ray.direction_sign[0]].x - ray.origin.x ) * ray.inv_direction.x;

	float t_y_min = ( box[ray.direction_sign[1]].y - ray.origin.y ) * ray.inv_direction.y;
	float t_y_max = ( box[1 - ray.direction_sign[1]].y - ray.origin.y ) * ray.inv_direction.y;

	float t_z_min = ( box[ray.direction_sign[2]].z - ray.origin.z ) * ray.inv_direction.z;
	float t_z_max = ( box[1 - ray.direction_sign[2]].z - ray.origin.z ) * ray.inv_direction.z;

	t0 = MAX( t_z_min, MAX( t_y_min, MAX( t_x_min, t0 ) ) );
	t1 = MIN( t_z_max, MIN( t_y_max, MIN( t_x_max, t1 ) ) );

	return ( t0 <= t1 ) && ( t1 > 0 );
}

bool BoxBoxIntersection( const AABB & box0, AABB & box1 )
{
	Vector3 half_d0 = ( box0.upper_bound() - box0.lower_bound() ) * static_cast<float>( 0.5 );
	Vector3 half_d1 = ( box1.upper_bound() - box1.lower_bound() ) * static_cast<float>( 0.5 );
	Vector3 d = half_d0 + half_d1;

	Vector3 t = box1.center() - box0.center();
	t.x = fabs( t.x );
	t.y = fabs( t.y );
	t.z = fabs( t.z );

	return ( ( t.x <= d.x ) || ( t.y <= d.y ) || ( t.z <= d.z ) );
}
