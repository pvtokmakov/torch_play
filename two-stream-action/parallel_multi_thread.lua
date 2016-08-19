require 'parallel'

function child()
    m = parallel.yield()

    for i = 1, 10 do
        print(m)
    end

    print 'sending stop'
    parallel.parent:send('stoped')
    print 'stop sent'
end

function parent()

    parallel.nfork(5)

    parallel.children:exec(child)

    for i = 1, 5 do
        print('join ' .. i)
        parallel.children[i]:join(i)
    end

    replies = parallel.children:receive()
    print(replies)

end

ok,err = pcall(parent)
if not ok then print(err) parallel.close() end