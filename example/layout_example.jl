using qgh

# butterfly continuous layout
function layout_example(::Val{:butterflycontinuous}; α = 0.0)
    points = Array{Float64, 2}([[] []])
    
    points_origin = [[43, 90], [55, 151], [76, 186], [92, 238], [116, 278], [134, 
    317], [138, 377], [127, 417], [148, 475], [182, 471], [211, 
    427], [231, 384], [254, 324], [271, 374], [294, 425], [315, 
    461], [355, 480], [365, 445], [365, 383], [356, 339], [386, 
    285], [408, 242], [431, 211], [446, 159], [461, 108], [431, 
    68], [401, 67], [364, 75], [330, 114], [299, 145], [270, 185], [238,
    196], [211, 151], [183, 117], [156, 87], [119, 60], [82, 51]]

    points_r = []
    
    for p in points_origin
        for x = -5:5:5, y = -5:5:5
            push!(points_r, [p[1]+x, p[2]+y])
        end
    end

    for i in points_origin
        points = vcat(points, [round(Int, (1-α)*i[1]+256*α) 512-i[2]]/512)
    end

    ContinuousLayout(points)
end

# circle continuous layout
function layout_example(::Val{:circle}; α = 0.0)
    points_origin = []
    for x in 0:0.02:1, y in 0:0.02:1
        if (x-0.5)^2 + (y-0.5)^2 <= 0.25^2
            push!(points_origin, [x, y])
        end
    end

    points = Array{Float64, 2}([[] []])
    for i in points_origin
        points = vcat(points, [(1.0 + 0.5 * α) * (i[1] - 0.5) + 0.5 (1 - 0.5 * α) * (i[2] - 0.5) + 0.5])
    end

    ContinuousLayout(points)
end