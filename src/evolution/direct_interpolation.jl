"""
evolution slm by matching the image of layout and slm.

Args:
    layout (Layout): the layout of traps.
    layout_end (Layout): the layout of traps at the end.
    slm (SLM): the initial SLM.
    algorithm (Algorithm): the algorithm to match the image.
    
Keyword Args:
    keypoint (Int): the number of key points between layout and layout_end.

Returns:
    layouts (Array{Layout}): the layouts of traps.
    slms (Array{SLM}): the SLMs.
"""
function evolution_slm_direct(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm; keypoints=5)
    points = layout.points
    Δx = (layout_end.points - layout.points) / keypoints # linear interpolation 

    slm, cost, B = match_image(layout, slm, algorithm)

    slms = [slm]
    layouts = [layout]
    for i in 1:keypoints
        println("Keypoints Step $i/$keypoints")
        layout = ContinuousLayout(points + i*Δx)
        slm, cost, B = match_image(layout, slm, B, algorithm)
        push!(slms, slm)
        push!(layouts, layout)
    end
    
    return layouts, slms
end