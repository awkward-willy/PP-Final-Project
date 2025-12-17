#pragma once

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>

#define SPACE_WIDTH 256
#define SPACE_HEIGHT 256

#define X_MIN ((-SPACE_WIDTH / 2))
#define Y_MIN ((-SPACE_HEIGHT / 2))
#define X_MAX ((SPACE_WIDTH / 2 - (SPACE_WIDTH % 2 == 0 ? 1 : 0)))
#define Y_MAX ((SPACE_HEIGHT / 2 - (SPACE_HEIGHT % 2 == 0 ? 1 : 0)))

struct Vector {
    /**
     * Constructor of Vector
     *
     * @param int location on the x axis
     * @param int location on the y axis
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector(int x, int y);

    /**
     * To compare two Vector objects (required for set)
     *
     * @param const Vector &other, another instance of Vector
     *
     * @return bool true if they are equal
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    bool operator==(const Vector& other) const;

    /**
     * To add two vectors
     *
     * @param const Vector& a reference to another vector
     *
     * @return Vector the sum
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator+(const Vector& other);

    /**
     * To add a vector to the current instance
     *
     * @param const Vector& a reference to another vector
     *
     * @return Vector the current vector
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator+=(const Vector& other);

    /**
     * To subtract two vectors
     *
     * @param const Vector& a reference to another vector
     *
     * @return Vector the subtraction
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator-(const Vector& other);

    /**
     * To subtract a vector to the current instance
     *
     * @param const Vector& a reference to another vector
     *
     * @return Vector the current vector
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator-=(const Vector& other);

    /**
     * To get a vector multiplied by a scalar value
     *
     * @param const int a scalar value
     *
     * @return Vector the product
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator*(const int factor);

    /**
     * to multiply the current vector
     *
     * @param const int a scalar value
     *
     * @return the current vector
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator*=(const int factor);

    /**
     * To get a vector divided by a scalar value
     *
     * @param const int a scalar value
     *
     * @return Vector the division
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator/(const int factor);

    /**
     * to divide the current vector
     *
     * @param const int a scalar value
     *
     * @return Vector the current vector
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Vector operator/=(const int factor);

    int x;
    int y;
};

/**
 * to define Vector hash (required in order to use set)
 *
 * @param const Vector &vector a Vector instance
 *
 * @author Thomas Marchand
 **/
template <>
struct std::hash<Vector> {
    std::size_t operator()(const Vector& vector) const noexcept;
};

struct Coordinates {
    /**
     * Constructor of Coordinates
     *
     * @param int location on the x axis
     * @param int location on the y axis
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Coordinates(int x, int y);

    /**
     * Get the square distance between two coordinates
     *
     * @param Coordinates another location
     *
     * @return double square distance to the next location
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    double square_distance_to(Coordinates location);

    /**
     * Get distance between two coordinates
     *
     * @param Coordinates another location
     *
     * @return double distance to the next location
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    double distance_to(double x, double y);

    /**
     * Get the square distance between two coordinates
     *
     * @param int x of the other location
     * @param int y other other location
     *
     * @return double square distance to the next location
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    double square_distance_to(double x, double y) const;
    std::vector<Coordinates> get_neighbors();

    /**
     * To check if Coordinates are equal
     *
     * @param const Coordinates &other, another instance of Coordinates
     *
     * @return bool true if they are equal
     *
     * @author Thomas Marchand
     **/
    bool operator==(const Coordinates& other) const;

    /**
     * To check if Coordinates are different
     *
     * @param const Coordinates &other, another instance of Coordinates
     *
     * @return bool true if they are different
     *
     * @author Thomas Marchand
     **/
    bool operator!=(const Coordinates& other) const;

    /**
     * To compare two coordinates
     *
     * @param const Coordinates &other, another instance of Coordinates
     *
     * @return bool true if the other is superior
     *
     * @author Thomas Marchand
     **/
    bool operator<(const Coordinates& other) const;

    /**
     * To add two Coordinates
     *
     * @param const Coordinates& a reference to another location
     *
     * @return Coordinates the sum
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Coordinates operator+(const Vector& other) const;

    /**
     * To subtract two Coordinates
     *
     * @param const Coordinates& a reference to another location
     *
     * @return Coordinates the difference
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Coordinates operator-(const Vector& other) const;

    int x;
    int y;
};

/**
 * to print Coordinates
 *
 * @param std::ostream &flux the current flux
 * @param const Coordinates &c the coordinates to display
 *
 * @author Thomas Marchand
 **/
std::ostream& operator<<(std::ostream& flux, const Coordinates& c);

template <>
/**
 * to define Coordinates hash (required in order to use set)
 *
 * @param const Coordinates &coordinates a Coordinate instance
 *
 * @author Thomas Marchand
 **/
struct std::hash<Coordinates> {
    std::size_t operator()(const Coordinates& coordinates) const noexcept;
};
