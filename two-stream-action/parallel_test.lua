require 'parallel'
require 'form_flow_batch'

function child()
    require 'form_flow_batch'

    math.randomseed(os.time())

    classes = parallel.parent:receive()
    videos = parallel.parent:receive()
    tree = parallel.parent:receive()
    labels = parallel.parent:receive()

    while true do
        local batch_ims, batch_labels = get_flow_batch()

        parallel.parent:send(batch_ims)
        parallel.parent:send(batch_labels)

        m = parallel.yield()
        if m == 'stop' then
            break
        end
    end
end

function parent()

    local videos, tree, classes, labels = load_dataset()

    local c = parallel.fork()
    c:exec(child)

    parallel.children:send(classes)
    parallel.children:send(videos)
    parallel.children:send(tree)
    parallel.children:send(labels)

    for i = 1, 5 do
        local ims = parallel.children:receive()
        print(ims)
        local batch_labels = parallel.children:receive()
        print(batch_labels)
        if i == 5 then
            c:join('stop')
        else
            c:join()
        end
    end
end

ok,err = pcall(parent)
if not ok then print(err) parallel.close() end