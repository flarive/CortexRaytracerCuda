#pragma once

#include <string>

#include "render_parameters.h"
//#include "../utilities/randomizer.h"

/**
 * The Singleton class defines the `GetInstance` method that serves as an
 * alternative to constructor and lets clients access the same instance of this
 * class over and over.
 */
class singleton
{

    /**
     * The Singleton's constructor should always be private to prevent direct
     * construction calls with the `new` operator.
     */

public:
    singleton(const render_parameters value) : value_(value)
    {

    }

    static singleton* singleton_;

    //std::string value_;

    render_parameters value_;
    //randomizer rnd_;

public:

    /**
     * Singletons should not be clonable.
     */
    singleton(singleton& other) = delete;
    /**
     * Singletons should not be assignable.
     */
    void operator=(const singleton&) = delete;
    /**
     * This is the static method that controls the access to the singleton
     * instance. On the first run, it creates a singleton object and places it
     * into the static field. On subsequent runs, it returns the client existing
     * object stored in the static field.
     */

     //static Singleton* GetInstance(const std::string& value);
    static singleton* getInstance();


    render_parameters value() const {
        return value_;
    }

    //randomizer& rnd() {
    //    return rnd_;
    //}
};


/**
 * Static methods should be defined outside the class.
 */
singleton* singleton::getInstance()
{
    /**
     * This is a safer way to create an instance. instance = new Singleton is
     * dangerous in case two instance threads wants to access at the same time
     */
     /* if (singleton_ == nullptr) {
          singleton_ = new Singleton(value);
      }*/
    return singleton_;
}