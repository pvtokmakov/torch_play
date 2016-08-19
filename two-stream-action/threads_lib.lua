local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local nthread = 1
local njob = 10

local Obj = {}

function Obj:temp()
    print('In function')
end


local pool = Threads(
    nthread,
    function()
        require 'torch'
    end,
    function(threadid)
        print('starting a new thread/state number ' .. threadid)
    end
)
print('Running jobs')

local jobdone = 0
for i=1,njob do
    pool:addjob(
        function()
            print(string.format('thread ID is %x', __threadid))
            return __threadid
        end,
        function (id)
        print('Woohoo! '.. id)
        end
    )
end

print('All jobs submited')

pool:synchronize()

print(string.format('%d jobs done', jobdone))

pool:terminate()

