/* -*-c++-*- */

#ifndef __INCLUDED_ARFF_DATA_H__
#define __INCLUDED_ARFF_DATA_H__
/**
 * @file arff_data.h
 * @brief Contains the 'ArffData' class
 */

#include <string>
#include <vector>
#include <map>

#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "arff_utils.h"
#include "arff_instance.h"
#include "arff_attr.h"

/// Class for hierarchical nominals
class ArffHierarchical {
private:
    std::vector<std::string> *items; // itemsets
    std::map<std::string, std::size_t> idx_map; // index mapping
    boost::adjacency_list<> *adj_list; // adjacent list
    bool is_tree_; // whether the hierarchy is a tree
    int32 num_items; // number of nominal items
public:
    /// Constructor and descructor
    ArffHierarchical();
    ~ArffHierarchical();
    /// Modify the members
    void parse_hierarchy(); // parse the hierarchy
    void set_items(std::vector<std::string> nominal_list); // input items
    /// accessor
    boost::adjacency_list<> *get_adjacency_list(); // get adj_list
    std::map<std::string, std::size_t> get_idx_map();
    std::size_t get_item_idx(std::string name);
    std::string get_item(std::size_t idx);
    std::vector<std::string> *get_item_list();
    bool is_tree();
    /// capacity
    int32 number_of_items();
protected:
    /// Parse a tree-style hierarchy
    void parse_tree();
    /// @todo: Parse a DAG-style hierarchy
    void parse_DAG();
    void set_items_DAG(std::vector<std::string> nominal_list);
    void set_items_tree(std::vector<std::string> nominal_list);

};

/** nominal values */
typedef std::vector<std::string> ArffNominal;

/** date formats */
typedef std::map<std::string, std::string> ArffDateFormat;

/**
 * @class ArffData arff_data.h
 * @brief Class to represent the data parsed from the ARFF files
 */
class ArffData{
public:
    /**
     * @brief Constructor
     */
    ArffData();

    /**
     * @brief Destructor
     */
    ~ArffData();

    /**
     * @brief Set the relation name
     * @param name the name
     */
    void set_relation_name(const std::string& name);

    /**
     * @brief Name of the relation for this ARFF file
     * @return the name
     */
    std::string get_relation_name() const;

    /**
     * @brief Number of attributes
     * @return number
     */
    int32 num_attributes() const;

    /**
     * @brief Add an attribute
     * @param attr attribute pointer
     *
     * Note that this pointer will be owned by this class from here onwards!
     */
    void add_attr(ArffAttr* attr);

    /**
     * @brief Get attribute pointer at the given location
     * @param idx location (Starts from 0)
     * @return pointer
     *
     * Note that this pointer will still be owned by this class!
     */
    ArffAttr* get_attr(int32 idx) const;

    /**
     * @brief Number of instances
     * @return number
     */
    int32 num_instances() const;

    /**
     * @brief Add an instance
     * @param inst instance pointer
     *
     * Note that this pointer will be owned by this class from here onwards!
     */
    void add_instance(ArffInstance* inst);

    /**
     * @brief Get instance pointer at the given location
     * @param idx location (Starts from 0)
     * @return pointer
     *
     * Note that this pointer will still be owned by this class!
     */
    ArffInstance* get_instance(int32 idx) const;

    /**
     * @brief Add a nominal value to the list
     * @param name name of the nominal list
     * @param val nominal value
     */
    void add_nominal_val(const std::string& name, const std::string& val);

    /**
     * @brief Get a nominal list
     * @param name name of the nominal list
     * @return list
     */
    ArffNominal get_nominal(const std::string& name);
    
    /**@todo
     * @brief Add a Hierarchical nominal value to the list
     * @param name name of the nominal list
     * @param val nominal value
     */
    void add_hierarchical_nominal(const std::string& name, const ArffHierarchical& val);

    /**
     * @brief Add a date format
     * @param name name of the date data
     * @param val date format
     */
    void add_date_format(const std::string& name, const std::string& val);

    /**
     * @brief Get a date format
     * @param name name of the date data
     * @return format
     */
    std::string get_date_format(const std::string& name);

    /**
     * @brief Add a hierarchical attribute
     * @param name name of the hierarchical attribute
     * @param val hierarchical value
     */
    void add_hierarchical_attr(const std::string& name, ArffHierarchical *attr);
    /**
     * @brief Get a hierarchical list
     * @param name name of the hierarchical list
     * @return list
     */
    ArffHierarchical *get_hierarchical(const std::string& name);

    /**
     * @brief Prepare an ARFF file from this object
     * @param file file to be written to
     */
    void write_arff(const std::string& file);


private:
    /**
     * @brief Cross check the attributes against the given instance
     * @param inst instance pointer
     */
    void _cross_check_instance(ArffInstance* inst);


    /** relation name */
    std::string m_rel;
    /** nominal values */
    std::map<std::string, ArffNominal> m_nominals;
    /** hierarchical values */
    std::map<std::string, ArffHierarchical *> m_hierarchicals;
    /** date formats */
    ArffDateFormat m_formats;
    /** number of attributes */
    int32 m_num_attrs;
    /** attributes */
    std::vector<ArffAttr *> m_attrs;
    /** number of instances */
    int32 m_num_instances;
    /** instances */
    std::vector<ArffInstance *> m_instances;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_DATA_H__
