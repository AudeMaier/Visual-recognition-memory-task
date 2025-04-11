using Random

function wsample(rng, set, weights)
    s = rand(rng)
    p = 0.
    for (w, x) in zip(weights, set)
        p += w
        p ≥ s && return x
    end
end
function chose_from_intervals(rng, intervals)
    lengths = @. last(intervals) - first(intervals)
    p = lengths ./ sum(lengths)
    a, b = intervals[wsample(rng, 1:length(intervals), p)]
    return rand(rng, a:b-1)
end
function new_interval(a, b)
    return b <= a ? (a, a-1) : (a, b)
end
function set_diff_intervals(a, b, intervals)
    if isempty(intervals)
        return [(a, b)]
    end
    sort!(intervals, by=first)
    complement = [new_interval(intervals[i][2], intervals[i+1][1]) for i in 1:length(intervals)-1]
    if intervals[1][1] > a
        complement = [(a, intervals[1][1])] ∪ complement
    end
    if intervals[end][2] < b
        complement = complement ∪ [(intervals[end][2], b)]
    end
    if complement[1][1] < a
        complement[1] = (a, complement[1][2])
    end
    if complement[end][2] > b
        complement[end] = (complement[end][1], b)
    end
    return complement
end

function create_dataset(; videos_tsv::String, n_samples::Int, clip_margin::Int=150, clip_length::Int=300, excluded_videos::Vector{String}=String[], seed::Int=0, rng=nothing, shuffle::Bool=true, fraction_videos::Float64=1.0, n_videos::Int=90, n_tests::Int=90)
    if rng === nothing
        rng = Xoshiro(seed)
    end
    # Get titles and lengths from videos_tsv
    titles = String[]
    lengths = Int[]
    open(videos_tsv, "r") do f
        for line in eachline(f)
            title, length = split(line, '\t')
            push!(titles, title)
            push!(lengths, parse(Int, length))
        end
    end
    # Remove excluded videos
    include_indices = [!(title in excluded_videos) for title in titles]
    titles = [titles[i] for i in 1:length(titles) if include_indices[i]]
    lengths = [lengths[i] for i in 1:length(lengths) if include_indices[i]]
    if shuffle
        indices = randperm(rng, length(titles))
        titles = [titles[i] for i in indices]
        lengths = [lengths[i] for i in indices]
    end
    titles = titles[1:floor(Int, fraction_videos * length(titles))]
    clips = zeros(Int, n_samples, n_videos + 2 * n_tests, 4)
    for i in 1:n_samples
        seen = Dict{Int, Vector{NTuple{2, Int}}}()
        possible_starts = Dict{Int, Vector{NTuple{2, Int}}}()
        seen_with_margins = Dict{Int, Vector{NTuple{2, Int}}}()
        for j in 1:n_videos
            video_index = rand(rng, 1:length(titles))
            if !haskey(seen, video_index)
                seen[video_index] = []
                possible_starts[video_index] = []
                seen_with_margins[video_index] = []
            end
            start = chose_from_intervals(rng, set_diff_intervals(0, lengths[video_index] - clip_length, seen[video_index]))
            clips[i, j, 1] = start
            clips[i, j, 2] = start + clip_length
            clips[i, j, 3] = video_index - 1
            clips[i, j, 4] = -1
            push!(seen[video_index], (start, start + clip_length))
            push!(possible_starts[video_index], (start - clip_length, start + clip_length))
            push!(seen_with_margins[video_index], (start - clip_margin, start + clip_length + clip_margin))
        end
        test_frames_seen = NTuple{2, Int}[]
        test_frames_unseen = NTuple{2, Int}[]
        unseen = Dict{Int, Vector{NTuple{2, Int}}}()
        for video_index in keys(seen_with_margins)
            unseen[video_index] = set_diff_intervals(0, lengths[video_index], seen_with_margins[video_index])
        end
        video_indices = collect(keys(seen))
        for j in 1:n_tests
            video_index = rand(rng, video_indices)
            test_frame = chose_from_intervals(rng, seen[video_index])
            push!(test_frames_seen, (video_index - 1, test_frame))
            video_index = rand(rng, video_indices)
            test_frame = chose_from_intervals(rng, unseen[video_index])
            push!(test_frames_unseen, (video_index - 1, test_frame))
        end
        test_frames = vcat(test_frames_seen, test_frames_unseen)
        labels = vcat(ones(Int, n_tests), zeros(Int, n_tests))
        indices = randperm(rng, 2 * n_tests)
        test_frames = [test_frames[i] for i in indices]
        labels = labels[indices]
        for j in 1:(2 * n_tests)
            clips[i, n_videos + j, 1] = test_frames[j][2]
            clips[i, n_videos + j, 2] = test_frames[j][2] + 1
            clips[i, n_videos + j, 3] = test_frames[j][1]
            clips[i, n_videos + j, 4] = labels[j]
        end
    end
    return Dict("clips" => clips, "titles" => titles)
end

