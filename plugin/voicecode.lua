-- Minimal Neovim bridge. Python is the sole writer for speech and code edits.
local job = nil
local root = vim.fn.fnamemodify(debug.getinfo(1, 'S').source:sub(2), ':p:h:h')
local function start()
  if job then
    vim.notify('Voicecode already running')
    return
  end
  local socket = vim.v.servername
  if socket == '' then
    socket = vim.fn.tempname() .. '.sock'
    vim.fn.serverstart(socket)
  end
  local cmd = vim.list_extend(vim.deepcopy(vim.g.voicecode_cmd or { 'uv', 'run', '--project', root, 'voicecode' }), { '--socket', socket })
  if vim.fn.executable(cmd[1]) == 0 then
    vim.notify('Voicecode: ' .. cmd[1] .. ' is not executable', vim.log.levels.ERROR)
    return
  end
  job = vim.fn.jobstart(cmd, {
    on_stderr = function(_, lines)
      for _, line in ipairs(lines) do
        if line ~= '' then vim.schedule(function() vim.notify('Voicecode: ' .. line, vim.log.levels.INFO) end) end
      end
    end,
    on_exit = function()
      job = nil
      vim.schedule(function() vim.notify('Voicecode stopped') end)
    end,
  })
  if job <= 0 then
    job = nil
    vim.notify('Could not start voicecode', vim.log.levels.ERROR)
  end
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
vim.keymap.set('n', '<Plug>(voicecode-toggle)', toggle, { desc = 'Voicecode: toggle listening' })
-- Default keymap unless disabled with `vim.g.voicecode_keymap = false` or the key is taken.
local lhs = vim.g.voicecode_keymap == nil and '<leader>vc' or vim.g.voicecode_keymap
if lhs and vim.fn.maparg(lhs, 'n') == '' then
  vim.keymap.set('n', lhs, '<Plug>(voicecode-toggle)', { desc = 'Voicecode: toggle listening' })
end
