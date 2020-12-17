#include "stdafx.h"

#include "volume.h"

using std::mt19937;
using std::uniform_real_distribution;

typedef mt19937                                     Engine;
typedef uniform_real_distribution<float>            Distribution;

auto uniform_generator = std::bind( Distribution( 0.0f, 1.0f ), Engine( 1 ) );

float Random( const float range_min, const float range_max )
{
	float ksi;

	#pragma omp critical ( random ) 
	{
		//ksi = static_cast< float >( rand() ) / ( RAND_MAX + 1 );
		ksi = uniform_generator();
	}

	return ksi  * ( range_max - range_min ) + range_min;
}

CellIndices::CellIndices( const int i, const int j, const int k )
{
	this->i = i;
	this->j = j;
	this->k = k;
}

Volume::Volume( const int width, const int height, const int n, const Vector3 & cell_size )
{
	assert( ( width > 1 ) && ( height > 1 ) && ( n > 1 ) );
	assert( CACHE_LINE_SIZE % kElement_size == 0 );
	assert( ( cell_size.x > 0 ) && ( cell_size.y > 0 ) && ( cell_size.z > 0 ) );

	width_ = width;
	height_ = height;
	n_ = n;

	{
		const int pad = ( ( kElement_size * width_ ) % CACHE_LINE_SIZE );
		width_step_ = ( ( kElement_size * width_ ) + pad ) / kElement_size;
	}

	data_ = new DATA_TYPE[size_t( width_step_ ) * height_ * n_];
	printf( "Volume data size: %0.1f MB\n", ( size_t( kElement_size * width_step_ ) * height_ * n_ ) / SQR( 1024.0 ) );

	cell_size_ = cell_size;
	half_volume_size_ = Vector3( cell_size_.x * ( n_ - 1 ), cell_size_.y * ( width_ - 1 ), cell_size_.z * ( height_ - 1 ) ) / 2;

	const Vector3 A = cell( 0, 0, 0 ).A();
	bounds_ = AABB( A, -A );
}

Volume::~Volume()
{
	SAFE_DELETE_ARRAY( data_ );
}

void Volume::Load( std::string & file_name_mask, const int first_slice_index, const int last_slice_index )
{
	printf( "Loading data...\n" );

	for ( int i = 0, n = 1; i < last_slice_index - first_slice_index + 1; ++i, ++n )
	{
		char file_name[128] = { 0 };
		sprintf( file_name, file_name_mask.c_str(), i + first_slice_index );

		cv::Mat slide_8u = cv::imread( std::string( file_name ), cv::IMREAD_GRAYSCALE );

		assert( slide_8u.cols == width_ );
		assert( slide_8u.rows == height_ );
		assert( n <= n_ );

		for ( int j = 0; j < slide_8u.cols; ++j )
		{
			for ( int k = 0; k < slide_8u.rows; ++k )
			{
				data_[offset( i, j, k )] = static_cast< DATA_TYPE >( slide_8u.at<uchar>( slide_8u.rows - 1 - k, j ) / 255.0 );
			}
		}

		printf( "\r%0.1f %%", ( 100.0 * n ) / n_ );
	}

	printf( "\rDone.   \n" );
}

void Volume::Generate()
{
	printf( "Generating data...\n" );

	cv::Mat slide_8u = cv::Mat( cv::Size( width_, height_ ), CV_32FC1 );

	for ( int i = 0; i < n_; ++i )
	{
		slide_8u.setTo( cv::Scalar( 0.0 ) );

		for ( int y = 0; y < slide_8u.rows; ++y )
		{
			for ( int x = 0; x < slide_8u.cols; ++x )
			{
				//slide_8u.at<uchar>( y, x ) = cv::saturate_cast< uchar >( 255 * ( 1.0 - sqrt( SQR( i - ( n_ - 1 ) * 0.5f ) + SQR( y - ( height_ - 1 ) * 0.5f ) + SQR( x - ( width_ - 1 ) * 0.5f ) ) / ( ( n_ - 1 ) * 0.5f + ( width_ - 1 ) * 0.5f*0 + ( height_ - 1 ) * 0.5*0 ) ) );
				slide_8u.at<float>( y, x ) = MAX( 0.0f, 1.0f * ( 1.0f - sqrt( SQR( i - ( n_ - 1 ) * 0.5f ) + SQR( y - ( height_ - 1 ) * 0.5f ) + SQR( x - ( width_ - 1 ) * 0.5f * 1 ) ) / ( ( n_ - 1 ) * 0.5f + ( width_ - 1 ) * 0.5f * 0 + ( height_ - 1 ) * 0.5f * 0 ) ) );
			}
		}

		for ( int j = 0; j < slide_8u.cols; ++j )
		{
			for ( int k = 0; k < slide_8u.rows; ++k )
			{
				//data_[offset( i, j, k )] = static_cast< DATA_TYPE >( slide_8u.at<uchar>( slide_8u.rows - 1 - k, j ) / 255.0 );
				data_[offset( i, j, k )] = static_cast< DATA_TYPE >( slide_8u.at<float>( slide_8u.rows - 1 - k, j ) );
			}
		}

		printf( "\r%0.1f %%", ( 100.0 * ( i + 1 ) ) / n_ );
	}

	printf( "\rDone.   \n" );
}

Cell Volume::cell( const CellIndices & indices ) const
{
	return cell( indices.i, indices.j, indices.k );
}

// [i, j, k] je index buòky
Cell Volume::cell(const int i, const int j, const int k) const
{
	assert((i >= 0) && (i < n_ - 1));
	assert((j >= 0) && (j < width_ - 1));
	assert((k >= 0) && (k < height_ - 1));

	DATA_TYPE rhos[4][4][4];

	// trilinear
	/*DATA_TYPE & rho_A = rhos[0];
	DATA_TYPE & rho_B = rhos[1];
	DATA_TYPE & rho_C = rhos[2];
	DATA_TYPE & rho_D = rhos[3];

	DATA_TYPE & rho_E = rhos[4];
	DATA_TYPE & rho_F = rhos[5];
	DATA_TYPE & rho_G = rhos[6];
	DATA_TYPE & rho_H = rhos[7];

	rho_A = data_[offset( i, j, k )];
	rho_B = data_[offset( i + 1, j, k )]; // B je A pro buòku [i + 1, j, k], atd.
	rho_C = data_[offset( i + 1, j + 1, k )];
	rho_D = data_[offset( i, j + 1, k )];

	rho_E = data_[offset( i, j, k + 1 )];
	rho_F = data_[offset( i + 1, j, k + 1 )];
	rho_G = data_[offset( i + 1, j + 1, k + 1 )];
	rho_H = data_[offset( i, j + 1, k + 1 )];*/

	// tricubic
	for (int x = 0; x < 4; x++)
		for (int y = 0; y < 4; y++)
			for (int z = 0; z < 4; z++)
			{
				auto o = Vector3(i + x - 1, j + y - 1, k + z - 1);
				auto n = o.x < 0 || o.x >= n_ || o.y < 0 || o.y >= width_ || o.z < 0 || o.z >= height_;
				rhos[x][y][z] = n ? 0 : data_[offset(o.x, o.y, o.z)];
			}

	const Vector3 A = Vector3(cell_size_.x * i, cell_size_.y * j, cell_size_.z * k) - half_volume_size_;

	return Cell((float*)rhos, A, A + cell_size_);
}

CellIndices Volume::cell_indices( const Vector3 & p ) const
{
	const Vector3 tmp = ( p - bounds_.lower_bound() ) / cell_size_;

	return CellIndices(
		MAX( 0, MIN( n_ - 2, static_cast< int >( floor( tmp.x ) ) ) ), // indexy první zasažené buòky
		MAX( 0, MIN( width_ - 2, static_cast< int >( floor( tmp.y ) ) ) ),
		MAX( 0, MIN( height_ - 2, static_cast< int >( floor( tmp.z ) ) ) ) );
}

void Volume::Traverse( Ray & ray, std::vector<CellHit> & traversed_cells )
{
	float t0 = 0;
	float t1 = REAL_MAX;

	traversed_cells.clear();

	if ( RayBoxIntersection( ray, bounds_, t0, t1 ) )
	{
		const Vector3 p = ray.eval( MIN( t0, t1 ) );

		ray.origin = p; // posuneme paprsek k objemu
		ray.t = REAL_MAX;

		CellIndices actual_cell_indices = cell_indices( p ); // indexy první zasažené buòky

		float t_0 = 0; // pøedpokládáme, že p leží na mezích objemu, první t_0 musí být tedy 0
		float t_1 = 0; // t_1 neumíme zatím urèit, takže taky 0		

		const CellIndices next_cell_indices = CellIndices( // zmìna indexu pøi prùchodu paprsku mezí buòky
			( ray.direction.x < 0 ) ? -1 : 1,
			( ray.direction.y < 0 ) ? -1 : 1,
			( ray.direction.z < 0 ) ? -1 : 1 );

		const Vector3 delta = ( cell_size_ / ray.direction ).Abs();	// parametrické vzdálenosti dvou následujících prùseèíkù vertikálních, resp. horizontálních, mezí bunìk		

		Vector3 t_max = Vector3( // parametrická souøadnice prvního prùseèíku s vertikální, resp. horizontální mezí objemu
			( bounds_[( ray.direction.x < 0 ) ? 1 : 0].x - p.x ) / ray.direction.x,
			( bounds_[( ray.direction.y < 0 ) ? 1 : 0].y - p.y ) / ray.direction.y,
			( bounds_[( ray.direction.z < 0 ) ? 1 : 0].z - p.z ) / ray.direction.z );
		
		t_max += delta * Vector3( // parametrická souøadnice prvního vnitøního prùseèíku s vertikální, resp. horizontální mezí bunìk
			( ray.direction.x < 0.0f ) ? float( n_ - 1 - actual_cell_indices.i ) : float( actual_cell_indices.i + 1 ),
			( ray.direction.y < 0.0f ) ? float( width_ - 1 - actual_cell_indices.j ) : float( actual_cell_indices.j + 1 ),
			( ray.direction.z < 0.0f ) ? float( height_ - 1 - actual_cell_indices.k ) : float( actual_cell_indices.k + 1 ) );

		bool is_inside = true; // pøíznak, že paprsek je stále uvnitø objemu

		while ( is_inside )
		{
			t_0 = t_1;

			switch ( t_max.SmallestComponent( true ) )
			{
			case 0: // tx je minimální
				t_1 = t_max.x;
				t_max.x += delta.x;
				//printf( "<%0.3f, %0.3f> in [%d, %d, %d]\n", t_0, t_1, actual_cell_indices.i, actual_cell_indices.j, actual_cell_indices.k ); // tady známe aktuální interval <t_0, t_1> nad (i, j, k)-tou buòkou
				if ( t_0 < t_1 ) traversed_cells.push_back( CellHit( actual_cell_indices, t_0, t_1 ) );
				actual_cell_indices.i += next_cell_indices.i; // protnuli jsme levou, resp. pravou, stìnu buòky, takže pokraèujeme levou, resp. pravou, buòkou
				if ( ( actual_cell_indices.i < 0 ) || ( actual_cell_indices.i >= n_ - 1 ) ) is_inside = false; // nebo is_inside &= ( ( i >= 0 ) && ( i < volume_i ) );
				break;

			case 1: // ty je minimální
				t_1 = t_max.y;
				t_max.y += delta.y;
				//printf( "<%0.3f, %0.3f> in [%d, %d, %d]\n", t_0, t_1, actual_cell_indices.i, actual_cell_indices.j, actual_cell_indices.k ); // tady známe aktuální interval <t_0, t_1> nad (i, j, k)-tou buòkou
				if ( t_0 < t_1 ) traversed_cells.push_back( CellHit( actual_cell_indices, t_0, t_1 ) );
				actual_cell_indices.j += next_cell_indices.j;
				if ( ( actual_cell_indices.j < 0 ) || ( actual_cell_indices.j >= width_ - 1 ) ) is_inside = false;
				break;

			case 2: // tz je minimální
				t_1 = t_max.z;
				t_max.z += delta.z;
				//printf( "<%0.3f, %0.3f> in [%d, %d, %d]\n", t_0, t_1, actual_cell_indices.i, actual_cell_indices.j, actual_cell_indices.k ); // tady známe aktuální interval <t_0, t_1> nad (i, j, k)-tou buòkou
				if ( t_0 < t_1 ) traversed_cells.push_back( CellHit( actual_cell_indices, t_0, t_1 ) );
				actual_cell_indices.k += next_cell_indices.k;
				if ( ( actual_cell_indices.k < 0 ) || ( actual_cell_indices.k >= height_ - 1 ) ) is_inside = false;
				break;
			}
		}
	}
	else
	{
		// nic, paprsek jde mimo objem
	}
}

// [i, j, k] je index buòky a chci ziskat offset do dat pro A
int Volume::offset( const int i, const int j, const int k ) const
{
	assert( ( i >= 0 ) && ( i < n_ ) );
	assert( ( j >= 0 ) && ( j < width_ ) );
	assert( ( k >= 0 ) && ( k < height_ ) );

	return ( width_step_ * height_ * i ) + ( j ) + ( ( height_ - 1 - k ) * width_step_ );
}

void Volume::Raycast(Camera& camera, const int samples)
{
	printf("Start ray casting...\n");

	cv::namedWindow("Render", cv::WINDOW_NORMAL);
	cv::moveWindow("Render", 0, 0);
	cv::Mat image = cv::Mat(cv::Size(camera.width(), camera.height()),
		CV_32FC3, cv::Scalar(0, 0, 0));
	cv::resizeWindow("Render", image.cols * 2, image.rows * 2);
	cv::imshow("Render", image);
	cv::waitKey(10);

	const int no_threads = omp_get_max_threads();
	omp_set_num_threads(no_threads);
	std::vector<CellHit>* traversed_cells = new std::vector<CellHit>[no_threads];

	const float ds = 1 / static_cast<float>(samples); // jittered sampling

	const double t0 = omp_get_wtime();

	int y;
	int no_rays = 0;
	const auto light = Vector3(0, 0, 1);

	// --------------- TASKS ----------------
	
	const int task = 3;

	// --------------- TASKS ----------------

#pragma omp parallel for schedule( dynamic, 12 ) default( none ) private( y ) shared( camera, no_rays, traversed_cells, image )
	for (y = 0; y < camera.height(); ++y)
		//y = 500;
	{
		const int tid = omp_get_thread_num();

		const float y0 = y + (ds - 1) * static_cast<float>(0.5);

		for (int x = 0; x < camera.width(); ++x)
			//int x = 500;
		{
			const float x0 = x + (ds - 1) * static_cast<float>(0.5);

			Vector3 pixel_color;
			float pixel_alpha = 0.0f;

			for (int ym = 0; ym < samples; ++ym)
			{
				for (int xm = 0; xm < samples; ++xm)
				{
					Vector3 sample_color;
					float sample_alpha = 0.0f;

					const float ksi1 = ((samples > 1) ? Random() : static_cast<float>(0.5));
					const float ksi2 = ((samples > 1) ? Random() : static_cast<float>(0.5));

					float cx = x0 + ((ksi1 - static_cast<float>(0.5)) + xm) * ds;
					float cy = y0 + ((ksi2 - static_cast<float>(0.5)) + ym) * ds;

					Ray ray = camera.GenerateRay(cx, cy);
#pragma omp atomic
					++no_rays;

					if (no_rays % 5000 == 0)
					{
						printf("\r%0.1f %%", (100.0 * no_rays) / (camera.width() * camera.height() * SQR(samples)));
						cv::imshow("Render", image);
						cv::waitKey(1);
					}

					Traverse(ray, traversed_cells[tid]);

					for (int i = 0; i < traversed_cells[tid].size(); ++i)
					{
						CellHit& cell_hit = traversed_cells[tid][i];
						Cell actual_cell = cell(cell_hit.indices);

						// TASK 0: maximum intensity projection (MIP)
						if (task == 0 || task == 1)
						{
							for (float t = cell_hit.t_0; t <= cell_hit.t_1; t += 0.1f)
							{
								sample_color.x = sample_color.y = sample_color.z = MAX(sample_color.x, actual_cell.Gamma(actual_cell.u(ray.eval(t))));
							}
							sample_alpha = 1.0f;
						}

						// TODO TASK 1: integrate along the ray
						if (task == 1) {
							const float value = actual_cell.Integrate(ray, cell_hit.t_0, cell_hit.t_1);
							sample_color += Vector3(value, value, value);
							sample_alpha = 0.01f;
						}

						// TODO TASK 2: find iso surface
						if (task == 2)
						{
							const float t_hit = actual_cell.FindIsoSurface(ray, cell_hit.t_0, cell_hit.t_1, 0.30f);
							if (t_hit > 0.0f)
							{
								const Vector3 p = ray.eval(t_hit);

								// TODO FOR POINT p: COMPUTE GRADIENT -> NORMAL -> LAMBERT SHADING + OMNI LIGHT -> SURFACE COLOR
								auto normal = actual_cell.GradGamma(actual_cell.u(p));
								normal.Normalize();

								auto lambert = MAX(normal.DotProduct(-light), 0);

								auto surface = Vector3(1, 1, 1) * lambert + Vector3(0.1, 0.1, 0.1);
								//surface = (normal * 0.5) + Vector3(0.5, 0.5, 0.5);

								sample_color = surface * _1_MPI;
								sample_alpha = 1.0f;

								break;
							}
						}

						// TODO TASK 3: DVR in front-to-back order
						if (task == 3)
						{
							if (actual_cell.FrontToBack(ray, cell_hit.t_0, cell_hit.t_1, &sample_color, &sample_alpha))
								break;
						}
					}

					pixel_color += sample_color * sample_alpha;
				}
			}

			pixel_color *= SQR(ds);
			image.at<cv::Vec3f>(y, x) = cv::Vec3f(powf(pixel_color.z, 1.0f / 2.2f), powf(pixel_color.y, 1.0f / 2.2f), powf(pixel_color.x, 1.0f / 2.2f));
		}
	}

	const double t1 = omp_get_wtime();

	printf("\r%0.3f s\n", (t1 - t0) * 1);

	cv::imshow("Render", image);
	image = image * 255;
	cv::imwrite("output.png", image);
	cv::waitKey( 0 );

	cv::destroyAllWindows();
}
