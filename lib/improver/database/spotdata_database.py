# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import iris
import iris.iterate
import iris.pandas
import numpy as np
import pandas as pd
from pandas import DataFrame


# For debugging
from profiler import profile
import pprint
import resource
class SpotDataProvider(list):
    def __init__(self):
        pass         
        




class SpotDatabase(object):
    
    def __init__(self, cubelist, primary_dim):
        
        """
        Initialise class.
        
              
        """
        
        self.cubelist  = cubelist
        self.primary_dim = primary_dim
        
        
        # For testing:
        import time
        testing_con = dict(realization = 0)
        if testing_con:
            self.cubelist = self.cubelist.extract(iris.Constraint(**testing_con))
        start = time.time()
        
        self.assert_similar()
        print time.time() - start
        
    def __repr__(self):
        
        """
        Representation of the instance.
        
        """
        
        
        return '<SpotDatabase: {}>'.format(self.primary_dim)
        
        
    def assert_similar(self):
        
        """
        Ensure that the dimensions and coordinates are shared between cubes.
        
        """
        cubelist = self.cubelist
        some_cube = self.cubelist[0]
        
        for cube in cubelist:
            for coord in cube.dim_coords:
                assert coord.is_compatible(some_cube.coord(coord.name()))
        
    
    def determine_dimensions(self, cube):
        
        """
        Determine the dimensions to collapse from the input cube.
        
        """
        
        dimensions = cube.dim_coords
        dim_names  = [dim.standard_name or dim.long_name for dim in dimensions]
        
        if [self.cols] + [self.rows] in dim_names:
            self.row.index = dim_names.index(self.row)
    @profile
    def to_dataframe(self):
        """
        Turn the input cubes into a 2-dimensional DataFrame object
        
        """
        
        cubes = self.cubelist
        rows  = self.primary_dim
        cols  = 'forecast_reference_time'
        coords   = [coord.standard_name or coord.long_name for coord in 
        
                                                    self.cubelist[0].coords()]
        required_dims  = ['time', 'index', 'wmo_site', 'forecast_period']
        ignored_coords = [item for item in coords if item not in required_dims]
        
        combined = DataFrame()
        for cube in cubes:
            print cube
            print ignored_coords
            for coord in ignored_coords:
                cube.remove_coord(coord)

            # Load one dataframe per station
            for c in cube.slices_over(1):
                print c
                #df = iris.pandas.as_data_frame(c)
                print c.data.shape
                df = DataFrame(np.diag(c.data), 
                               index=c.coord('time'),
                               columns=c.coord('forecast_period').points)  
                
                df['station_id'] = c.coord('wmo_site').points[0]
        
                combined = pd.concat([df, combined])
                print combined
        return combined
        
        
        
        
        datapoints = [cube.data.size for cube in self.cubelist]
        print '{} datapoints/cube = {} total'.format(datapoints,sum(datapoints))
        
        for cube in cubes:
            ignored_coord_dims = [item for item in ignored_coords if item not in cube.dim_coords]
            print cube
            for c in cube.slices_over('time'):
                df = iris.pandas.as_data_frame(c, copy=False)
                print df
        
        return
        primary_key_iterator = self.determine_range(rows)
        for row_val in sorted(primary_key_iterator, key=lambda x: x.points):
            print row_val
            records = []
            
            # Constrain the cubes by primary_dim
            constraint = iris.Constraint(**{rows : row_val.points})
            selection  = self.cubelist.extract(constraint)
            
            #selection = self.cubelist
            record  = dict()
            for cube in selection:
                print cube
                df = iris.pandas.as_data_frame(cube, copy=False)
                print df
            return
            for iterator in iris.iterate.izip(*selection):
                print iterator[0].data
                df = iris.pandas.as_data_frame(cube, copy=False)
                print df
                for cube in iterator:
                    dictionary = cube.coords()
                    record = {coord.standard_name or
                              coord.long_name : coord.points[0]
                              for coord in [cube.coord(c) for c in cols]}
                    records.append(record)
            print 'Resource usage {} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024. ),
            print ' {}/{} Records'.format(len(records),len(df)),
            print row_val
            new_df = DataFrame.from_dict(records)
            df =  pd.concat([df, new_df])
        return df
        #df.set_index('time', 'wmo_index')
        
        #frames = [DataFrame.from_records(cube.data, columns=cols)
        #          for cube in selection]
        #print [c.data.shape for c in selection]
        
    def to_sql(self, dataframe):
        dataframe.to_sql()
    
    def determine_range(self, dimension):
    
        """
        Determine the unique values of the dimension over which to unroll into 
        primary key rows in the table.
        
        """
        
        unique_values = set()
        for cube in self.cubelist:
            for val in cube.coord(dimension):
                unique_values.add(val)
                
        return unique_values
        
