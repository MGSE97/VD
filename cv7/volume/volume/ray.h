#ifndef RAY_H_
#define RAY_H_

#include "vector3.h"

/*! \struct Ray
\brief Paprsek.

\f$\mathbf{r}(t) = O + \hat{\mathbf{d}}t\f$

\author Tomáš Fabián
\version 1.0
\date 2011-2013
*/
struct Ray
{
	Vector3 origin; /*!< Poèátek paprsku. */
	Vector3 direction; /*!< Smìrový vektor (jednotkový). */
	Vector3 inv_direction; /*!< Invertovaný smìrový vektor. */
	float t; /*!< Reálný parametr \f$tf$. */	
	char direction_sign[3]; /*!< Znaménko smìrového vektoru. */

	//! Specializovaný konstruktor.
	/*!
	Inicializuje paprsek podle zadaných hodnot a jdoucí do nekoneèna.

	\param origin poèátek.
	\param direction jednotkový smìr.
	*/
	Ray( const Vector3 & origin, const Vector3 & direction, const float bounce, const float env_ior )
	{
		this->origin = origin;
		set_direction( direction );

		this->origin += bounce * this->direction;

		t = REAL_MAX;				
	}

	//! Vypoète \f$\mathbf{r}(t)\f$.
	/*!
	\return Souøadnice bodu na paprsku pro zadaný parametr \f$t\f$.
	*/
	Vector3 eval( const float t )
	{
		return origin + direction * t;
	}

	//! Vypoète \f$r(t)\f$.
/*!
\return True je-li \a t menší než .
*/
	bool closest_hit( const float t )
	{

		bool hit_confirmed = false;

		//#pragma omp critical ( make_hit )
		{
			if ( ( t < this->t ) && ( t > 0 ) )
			{
				this->t = t;
				hit_confirmed = true;				
			}
		}

		return hit_confirmed;
	}

	bool is_hit() const
	{
		return ( ( t > 0 ) && ( t < REAL_MAX ) /*&& ( target != NULL )*/ );
	}

	void set_direction( const Vector3 & direction )
	{
		this->direction = direction;
		this->direction.Normalize();
		inv_direction = Vector3( 1 / this->direction.x, 1 / this->direction.y, 1 / this->direction.z );
		direction_sign[0] = static_cast< char >( inv_direction.x < 0 ); // 0 pro <0,+inf> a 1 pro <-inf,0)
		direction_sign[1] = static_cast< char >( inv_direction.y < 0 );
		direction_sign[2] = static_cast< char >( inv_direction.z < 0 );
	}
};

#endif
