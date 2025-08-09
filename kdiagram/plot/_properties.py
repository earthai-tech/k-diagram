


TDG_DIRECTIONS= { 
    "-1":{ 
        "CORR_POS": {
                "NE": [
                    (1., 0.5),
                    {"rotation_mode": "anchor", "rotation": 90},
                ],
                "SW":[
                    (0., 0.5),
                    {"rotation_mode": "anchor", "rotation": 90},
                ],
                
                "SE": [
                    (0.5, 0.05), {"rotation": "horizontal"},
                ],
                
                "NW": [
                    (0.5, 1.), {"rotation": "horizontal"}
                ],
    
                "W": [
                    (0., 0.75),
                    {"rotation_mode": "anchor", "rotation": 60},
                ],
                "E": [
                    (1.0, 0.5),
                    {"rotation_mode": "anchor", "rotation": 60},
                ],
                "N": [
                    (1.0, 0.75),
                    {"rotation_mode": "anchor", "rotation": -60},
                ],
                "S": [
                    (0.0, 0.5),
                    {"rotation_mode": "anchor", "rotation": -60},
                ],
        }, 
        "STD_POS" : {
                "NE":  [
                    (0.25, 0.75),
                    {"rotation_mode": "anchor", "rotation": 45},
                    ], 
                "SW":[
                    (0.75, 0.25),
                   {"rotation_mode": "anchor", "rotation": 45},
                    ], 
                
                "SE": [
                    (0.75, 0.70), 
                    {"rotation_mode": "anchor", "rotation": -45}
                    ], 
                
                "NW":[
                    (0.15, .35),
                    {"rotation_mode": "anchor", "rotation": -45},
                    ], 
                "W": [
                    (0.5, -0.1),{"rotation": "horizontal"},
                ],
                "E": [
                    (0.5, 1.05),{"rotation": "horizontal"},
                ],
                "N": [
                    (-0.1, 0.5),{"rotation": "vertical"},
                ],
                "S": [
                    (1.1, 0.5),{"rotation": "vertical"},
                ],
            }
    
        },
    
    "1":{ 
        "CORR_POS" : {
              "NW":[
                (0., .5),
                {"rotation_mode": "anchor", "rotation": 90},
                ], 
            ###
            "NE":  [
                (0.5, 0.95), {"rotation": "horizontal"}, 
                ], 
            ##
            "SW":[
                (0.5, 0.05), {"rotation": "horizontal"},
                ], 
            
            "SE": [
                (1.1, 0.5), {"rotation": "vertical"}
                ], 
            
    
            "W": [
                (0.20, 0.25),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "E": [
                (0.8, .75),
                {"rotation_mode": "anchor", "rotation": -45},
            ##            
            ],
            "N": [
                (.1, 0.7),
                {"rotation_mode": "anchor", "rotation": 45},
                
               
            ],
            ##
            "S": [
                (.85, 0.3),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
        },
    
    "STD_POS" : {
            #
            "NW": [
                (0.65, 0.85), 
                {"rotation_mode": "anchor", "rotation": -45},
            ],
    
            "NE": [
                (0.90, 0.40),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            
            "SW":[
                (0.15, 0.65),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            "SE": [
                (0.25, 0.25), 
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "W": [
                (0.5, 1.05),{"rotation": "horizontal"},
            ##
            ],
            "E": [
                (.5, -0.1),{"rotation": "horizontal"},
            ],
            ##
            "N": [
                (1.1, 0.5),{"rotation": "vertical"},
                
            ],
            
            "S": [
                (-.1, 0.5),{"rotation": "vertical"},
            ],
        }
    
   } 
    
} 
