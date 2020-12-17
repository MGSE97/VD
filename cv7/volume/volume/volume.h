#ifndef VOLUME_H_
#define VOLUME_H_

#include "vector3.h"
#include "matrix4x4.h"
#include "aabb.h"
#include "ray.h"
#include "intersection.h"
#include "cell.h"
#include "camera.h"

float Random( const float range_min = 0.0f, const float range_max = 1.0f );

#define CACHE_LINE_SIZE 64

/*! \struct CellIndices
\brief Struktura adresy buòky v rámci celého objemu.

\author Tomáš Fabián
\version 1.0
\date 2015-2020
*/
struct CellIndices
{
public:
	CellIndices()
	{
		i = -1;
		j = -1;
		k = -1;
	}

	CellIndices( const int i, const int j, const int k );

	int i;
	int j;
	int k;
};

/*! \struct CellHit
\brief Struktura popisující zasaženou buòku (její index v rámci celého objemu a parametrické body vstupu a výstupu paprsku).

\author Tomáš Fabián
\version 1.0
\date 2015-2020
*/
struct CellHit
{
public:
	CellHit( CellIndices & indices, const float t_0, const float t_1 )
	{
		assert( ( t_0 >= 0.0f ) && ( t_1 >= t_0 ) );
		// pokuï je t_0 == t_1, pak to prochází jen rohem buòky a ta mùže být pøeskoèena

		this->indices = indices;
		this->t_0 = t_0;
		this->t_1 = t_1;

		f = 0.0f;
	}

	CellIndices indices;

	float t_0;
	float t_1;

	float f;
};

/*! \class Volume
\brief Tøída pro pøímou vizualizaci skalárních objemových dat.

\author Tomáš Fabián
\version 1.0
\date 2015-2020
*/
class Volume
{
public:
	Volume( const int width, const int height, const int n, const Vector3 & cell_size );
	~Volume();

	void Load( std::string & file_name_mask, const int first_slice_index, const int last_slice_index );
	
	void Generate();		

	Cell cell( const CellIndices & indices ) const;
	Cell cell( const int i, const int j, const int k ) const;

	CellIndices cell_indices( const Vector3 & p ) const; // vrátí indexy buòky, ve které leží bod p
	
	void Traverse( Ray & ray, std::vector<CellHit> & traversed_cells );

	void Raycast( Camera & camera, const int samples = 1 );
	
protected:
	int offset( const int i, const int j, const int k ) const;	

private:
	int width_; // horizontální rozlišení zdrojového snímku [px]; osa y
	int height_; // vertikální rozlišení zdrojového snímku [px]; osa -z
	int n_; // poèet øezù [-]; osa x

	int width_step_; // poèet reálných hodnot na jednom øádku datové matice [-]
	
	Vector3 cell_size_; // rozmìry buòky [m]
	Vector3 half_volume_size_; // rozmìry pùlky celého objemu [m]

	DATA_TYPE * data_; // datová matice s hustotami o rozmìrech (width_step_, height_, n_) a v tomto poøadí jsou také uloženy jednotlivé hodnoty

	AABB bounds_;

	static const int kElement_size = sizeof( DATA_TYPE );
};

#endif
