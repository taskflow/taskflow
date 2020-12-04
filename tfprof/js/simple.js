const simple = [
{
  "executor": 3,
  "data": [
    {
      "worker": 1,
      "level": 0,
      "data": [
        {
          "span": [2,7],
          "name": "t1",
          "type": "static"
        },
        {
          "span": [9,13],
          "name": "t2",
          "type": "subflow"
        },
        {
          "span": [15,20],
          "name": "t3",
          "type": "cudaflow"
        },
        {
          "span": [22,27],
          "name": "t4",
          "type": "condition"
        },
        {
          "span": [29,34],
          "name": "t5",
          "type": "module"
        },
        {
          "span": [36,40],
          "name": "t6",
          "type": "static"
        }
      ]
    },
    {
      "worker": 2,
      "level": 0,
      "data":[
        {
          "span": [4,5],
          "name": "t7",
          "type": "subflow"
        },
        {
          "span": [9,10],
          "name": "t8",
          "type": "cudaflow"
        },
        {
          "span": [15,16],
          "name": "t9",
          "type": "condition"
        },
        {
          "span": [19,20],
          "name": "t10",
          "type": "module"
        },
        {
          "span": [22,23],
          "name": "t11",
          "type": "static"
        },
        {
          "span": [26,27],
          "name": "t12",
          "type": "subflow"
        },
        {
          "span": [29,30],
          "name": "t13",
          "type": "cudaflow"
        },
        {
          "span": [33,34],
          "name": "t14",
          "type": "condition"
        },
        {
          "span": [36,37],
          "name": "t15",
          "type": "module"
        }
      ]
    },
    {
      "worker": 3,
      "level": 0,
      "data":[
        {
          "span": [4,5],
          "name": "t16",
          "type": "condition"
        },
        {
          "span": [9,13],
          "name": "t17",
          "type": "module"
        },
        {
          "span": [15,20],
          "name": "t18",
          "type": "static"
        },
        {
          "span": [22,27],
          "name": "t19",
          "type": "cudaflow"
        },
        {
          "span": [29,30],
          "name": "t21",
          "type": "condition"
        },
        {
          "span": [33,34],
          "name": "t22",
          "type": "module"
        },
        {
          "span": [36,40],
          "name": "t23",
          "type": "condition"
        }
      ]
    }
  ]
},
{
  "executor": 1,
  "data": [
    {
      "worker": 1,
      "level": 0,
      "data": [
        {
          "span": [4,5],
          "name": "t24",
          "type": "cudaflow"
        },
        {
          "span": [9,10],
          "name": "t25",
          "type": "module"
        },
        {
          "span": [15,16],
          "name": "t26",
          "type": "cudaflow"
        },
        {
          "span": [22,23],
          "name": "t27",
          "type": "cudaflow"
        },
        {
          "span": [25,26],
          "name": "t28",
          "type": "static"
        },
        {
          "span": [29,30],
          "name": "t29",
          "type": "static"
        },
        {
          "span": [33,34],
          "name": "t30",
          "type": "static"
        },
        {
          "span": [36,37],
          "name": "t31",
          "type": "cudaflow"
        }
      ]
    },
    {
      "worker": 2,
      "level": 0,
      "data": [
        {
          "span": [4,5],
          "name": "t32",
          "type": "module"
        },
        {
          "span": [9,10],
          "name": "t33",
          "type": "static"
        },
        {
          "span": [15,16],
          "name": "t34",
          "type": "condition"
        },
        {
          "span": [22,23],
          "name": "t35",
          "type": "static"
        },
        {
          "span": [26,27],
          "name": "t36",
          "type": "module"
        },
        {
          "span": [29,34],
          "name": "t37",
          "type": "module"
        },
        {
          "span": [36,37],
          "name": "t38",
          "type": "module"
        }
      ]
    }
  ]
}
]
