#ifndef CAMERA_H_
#define CAMERA_H_

#include "vector3.h"
#include "matrix4x4.h"
#include "ray.h"

/*! \class Camera
\brief Implementace jednoduché kamery.

\author Tomáš Fabián
\version 1.0
\date 2011-2012
*/
class Camera
{
public:

	Camera()
	{		
		Reset();
	}

	Camera( const Vector3 & view_from, const Vector3 & view_at, const int width, const int height, const float fov_y )
	{
		width_ = width;
		height_ = height;

		up_ = Vector3( 0, 0, 1 );

		view_from_ = view_from;
		view_at_ = view_at;

		d_ = 1;
		fov_y_ = fov_y;		
		pixel_size_ = 2 * d_ * tan( fov_y_ / 2 ) / height_;				

		BuildViewMatrix();		
	}

	~Camera()
	{

	}

	void Reset()
	{
		width_ = 1024;
		height_ = 768;		
		
		alt_ = DEG2RAD( 70 );
		azim_ = DEG2RAD( 40 );
		fov_y_ = DEG2RAD( 45 );

		d_ = 1;
		pixel_size_ = 2 * d_ * tan( fov_y_ / 2 ) / height_;				

		zoom_ = 1;

		UpdateDirection();

		view_from_ = Vector3( -0.5f, -1.3f, 0.65f );
		view_at_ = Vector3( 0.0f, 0.0f, 0.25f );

		BuildViewMatrix();

		near_ = 0.1f;
		far_ = 30.0f;

		view_ = Matrix4x4();
		projection_ = Matrix4x4();
	}

	float aspect_ratio()
	{
		return width_ / static_cast<float>( height_ );
	}

	void set_width( const int width )
	{
		assert( width > 0 );

		width_ = width;
	}

	void set_height( const int height )
	{
		assert( height > 0 );

		height_ = height;
	}

	float fov_y()
	{
		return fov_y_;
	}

	int width()
	{
		return width_;
	}

	int height()
	{
		return height_;
	}

	Vector3 eye() const
	{
		return view_from_;		
	}	

	float zoom_; /*!< Vzdálenost oka od center_. */	

	void UpdateDirection()
	{				
		view_dir_.x = zoom_ * sinf( alt_ ) * sinf( azim_ );
		view_dir_.y = zoom_ * sinf( alt_ ) * cosf( azim_ );
		view_dir_.z = zoom_ * cosf( alt_ );
	}	

	Ray GenerateRay( const float sx, const float sy );

	Vector3 axis_x() const
	{
		return axis_x_;
	}

	Vector3 axis_y() const
	{
		return axis_y_;
	}

	Vector3 axis_z() const
	{
		return axis_z_;
	}

	void BuildViewMatrix()
	{		
		axis_z_ = view_from_ - view_at_;
		axis_z_.Normalize();
		axis_x_ = up_.CrossProduct( axis_z_ );
		axis_x_.Normalize(); // renormalizace je tady nutná
		axis_y_ = axis_z_.CrossProduct( axis_x_ );
		axis_y_.Normalize(); // renormalizace je tady nutná

		view_.set( 0, 0, axis_x_.x );
		view_.set( 0, 1, axis_y_.x );
		view_.set( 0, 2, axis_z_.x );
		view_.set( 0, 3, view_from_.x );

		view_.set( 1, 0, axis_x_.y );
		view_.set( 1, 1, axis_y_.y );
		view_.set( 1, 2, axis_z_.y );
		view_.set( 1, 3, view_from_.y );

		view_.set( 2, 0, axis_x_.z );
		view_.set( 2, 1, axis_y_.z );
		view_.set( 2, 2, axis_z_.z );
		view_.set( 2, 3, view_from_.z );

		/*view_.set( 3, 0, 0 );
		view_.set( 3, 1, 0 );
		view_.set( 3, 2, 0 );
		view_.set( 3, 3, 1 );*/

		view_.EuclideanInverse();
	}

	void BuildProjectionMatrix()
	{
		assert( ( near_ > 0 ) && ( far_ > near_ ) );

		const float height_half = near_ * tan( fov_y_ / 2 );
		const float width_half = height_half * aspect_ratio();

		projection_.set( 0, 0, near_ / width_half );
		projection_.set( 1, 1, near_ / height_half );
		projection_.set( 2, 2, ( far_ + near_ ) / ( near_ - far_ ) );
		projection_.set( 2, 3, 2 * far_ * near_ / ( near_ - far_ ) );
		projection_.set( 3, 2, -1 );
		projection_.set( 3, 3, 0 );
	}

	Matrix4x4 ModelViewMatrix( const Matrix4x4 & model )
	{
		return view_ * model;
	}

	Matrix4x4 ModelViewProjectionMatrix( const Matrix4x4 & model )
	{
		return projection_ * view_ * model;
	}

	Matrix4x4 ModelViewNormalMatrix( const Matrix4x4 & model )
	{
		Matrix4x4 normal = view_ * model;
		normal.EuclideanInverse();
		normal.Transpose();

		return normal;
	}

	Matrix4x4 & view_matrix()
	{
		return view_;
	}

	Matrix4x4 & projection_matrix()
	{
		return projection_;
	}

private:
	int width_; /*!< Šíøka obrazu [px]. */
	int height_; /*!< Výška obrazu [px]. */

	Vector3 view_dir_; /*!< Opaèný smìr pohledu. Vypoèítává se na základì zoom_, alt_ a azim_. */
	Vector3 center_; /*!< Bod, na který se díváme. */

	float alt_; /*!< Úhlová výška v radiánech. */
	float azim_; /*!< Azimutální úhel v radiánech. */

	float fov_y_; /*!< Vertikální zorné pole [rad]. */
	float d_; /*!< Vzdálenost promítací roviny od støedu promítání (oka) [m]. */
	float pixel_size_; /*!< Velikost 1 pixelu [m/px]. */

	Vector3 axis_x_; /*!< Bázový vektor souøadného systému reprezentující x-ovou osu. */
	Vector3 axis_y_; /*!< Bázový vektor souøadného systému reprezentující y-ovou osu. */
	Vector3 axis_z_; /*!< Bázový vektor souøadného systému reprezentující z-ovou osu. */	

	// --- OpenGL ---
	Vector3 up_;
	Vector3 view_from_;
	Vector3 view_at_;	

	float near_;
	float far_;

	Matrix4x4 view_; /*!< Matice pøechodu z world-space do eye-space. */
	Matrix4x4 projection_;
};

#endif
