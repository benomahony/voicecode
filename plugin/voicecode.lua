-- Minimal Neovim bridge. Python is the sole writer for speech and code edits.
local job = nil
local root = vim.fn.fnamemodify(debug.getinfo(1, 'S').source:sub(2), ':p:h:h')
local log_path = vim.fn.stdpath('log') .. '/voicecode.log'
local api_key = nil -- Kept for this Neovim session only, never written to disk.
local levels = { info = vim.log.levels.INFO, error = vim.log.levels.ERROR, auth = vim.log.levels.ERROR }

local function notify(msg, level)
  vim.schedule(function() vim.notify(msg, level or vim.log.levels.INFO, { title = 'Voicecode' }) end)
end

-- Job callbacks deliver chunks; hold back the trailing partial line until it completes.
local function by_line(on_line)
  local pending = ''
  return function(_, data)
    data[1] = pending .. data[1]
    pending = table.remove(data)
    for _, line in ipairs(data) do on_line(line) end
  end
end

-- Only the default OpenAI provider is handled here; other providers read their own env vars.
local function key_env()
  local model = vim.env.VOICECODE_MODEL or 'openai:'
  if not vim.startswith(model, 'openai:') or (vim.env.OPENAI_API_KEY or '') ~= '' then return {} end
  if not api_key and vim.g.voicecode_api_key_cmd then
    local result = vim.system(vim.g.voicecode_api_key_cmd, { text = true }):wait()
    if result.code == 0 then api_key = vim.trim(result.stdout) end
    if result.code ~= 0 then notify('vim.g.voicecode_api_key_cmd failed: ' .. vim.trim(result.stderr), vim.log.levels.WARN) end
  end
  api_key = (api_key and api_key ~= '') and api_key or vim.fn.inputsecret('OpenAI API key: ')
  if api_key == '' then api_key = nil end
  return api_key and { OPENAI_API_KEY = api_key } or nil
end

local function backend()
  return vim.deepcopy(vim.g.voicecode_cmd or { 'uv', 'run', '--project', root, 'voicecode' })
end

local function start()
  if job then return notify('Already running') end
  local env = key_env()
  if not env then return notify('No OpenAI API key; not started', vim.log.levels.WARN) end
  local socket = vim.v.servername ~= '' and vim.v.servername or vim.fn.serverstart(vim.fn.tempname() .. '.sock')
  local cmd = vim.list_extend(backend(), { '--socket', socket })
  if vim.g.voicecode_device then vim.list_extend(cmd, { '--device', vim.g.voicecode_device }) end
  if vim.fn.executable(cmd[1]) == 0 then return notify(cmd[1] .. ' is not executable', vim.log.levels.ERROR) end
  local log = assert(io.open(log_path, 'w'))
  local function write(line) log:write(line, '\n'); log:flush() end
  job = vim.fn.jobstart(cmd, {
    env = next(env) and env or nil,
    on_stdout = by_line(function(line)
      local level, msg = line:match('^(%a+)\t(.*)$')
      if level == 'auth' then api_key = nil end
      write(line)
      if levels[level] then notify(msg, levels[level]) end
    end),
    on_stderr = by_line(write),
    on_exit = function(id, code)
      if job == id then job = nil end -- A restart may already own `job`.
      log:close()
      if code == 0 or code == 130 or code == 143 then return notify('Stopped') end
      notify(('Exited with code %d; :VoicecodeLog has the details'):format(code), vim.log.levels.ERROR)
    end,
  })
  if job <= 0 then
    job = nil
    log:close()
    return notify('Could not start ' .. cmd[1], vim.log.levels.ERROR)
  end
  notify('Starting (the first run installs dependencies)')
end
local function stop()
  if job then vim.fn.jobstop(job); job = nil end
end
local function toggle()
  if job then stop() else start() end
end
vim.api.nvim_create_user_command('VoicecodeStart', start, {})
vim.api.nvim_create_user_command('VoicecodeStop', stop, {})
vim.api.nvim_create_user_command('VoicecodeToggle', toggle, {})
vim.api.nvim_create_user_command('VoicecodeLog', function() vim.cmd.split(vim.fn.fnameescape(log_path)) end, {})

local function input_devices()
  local result = vim.system(vim.list_extend(backend(), { '--list-devices' }), { text = true }):wait()
  local ok, names = pcall(vim.json.decode, result.stdout or '')
  if result.code ~= 0 or not ok then
    notify('Could not list input devices; :VoicecodeLog has the details', vim.log.levels.ERROR)
    return {}
  end
  return names
end

local function use_device(name)
  vim.g.voicecode_device = name ~= '' and name or nil
  notify('Input: ' .. (vim.g.voicecode_device or 'system default'))
  if job then stop(); start() end
end

vim.api.nvim_create_user_command('VoicecodeDevice', function(cmd)
  if cmd.args ~= '' then return use_device(cmd.args) end
  local names = input_devices()
  if #names == 0 then return end
  table.insert(names, 1, '')
  vim.ui.select(names, {
    prompt = 'Voicecode input device',
    format_item = function(name)
      return name == '' and 'System default' or (name == vim.g.voicecode_device and name .. ' (current)' or name)
    end,
  }, function(name)
    if name then use_device(name) end
  end)
end, {
  nargs = '?',
  desc = 'Choose the microphone or aggregate device voicecode listens to',
  complete = function(lead)
    return vim.tbl_filter(function(name) return name:lower():find(lead:lower(), 1, true) end, input_devices())
  end,
})
vim.keymap.set('n', '<Plug>(voicecode-toggle)', toggle, { desc = 'Voicecode: toggle listening' })
-- Default keymap unless disabled with `vim.g.voicecode_keymap = false` or the key is taken.
local lhs = vim.g.voicecode_keymap == nil and '<leader>vc' or vim.g.voicecode_keymap
if lhs and vim.fn.maparg(lhs, 'n') == '' then
  vim.keymap.set('n', lhs, '<Plug>(voicecode-toggle)', { desc = 'Voicecode: toggle listening' })
end
